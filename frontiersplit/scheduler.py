"""FrontierSplit Concurrent Request Scheduler.

Manages multi-request queueing, continuous micro-batch interleaving across the
pipeline stages, streaming Server-Sent Events (SSE), and pipeline bubble telemetry.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional
import requests
from pydantic import BaseModel, Field

from frontiersplit.protocol import ActivationPacket, GenerationResponse


class ScheduledRequest:
    """Tracks state and history for an active chat completion request."""

    def __init__(
        self,
        request_id: str,
        model: str,
        prompt_text: str,
        prompt_tokens: List[int],
        max_tokens: int = 64,
        temperature: float = 0.7,
        stream: bool = False,
    ):
        self.request_id = request_id
        self.model = model
        self.prompt_text = prompt_text
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stream = stream

        self.created_at = time.time()
        self.current_step = 0
        self.is_prefill = True
        self.input_tokens = list(prompt_tokens) if prompt_tokens else [1]

        self.generated_tokens: List[int] = []
        self.generated_chunks: List[str] = []
        self.is_finished: bool = False
        self.finish_reason: Optional[str] = None
        self.error: Optional[str] = None

        self.done_event = asyncio.Event()
        self.stream_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

        self.step_latencies: List[float] = []
        self.stage_timings: List[Dict[str, float]] = []

    def to_chat_completion_response(self) -> Dict[str, Any]:
        """Format as standard OpenAI ChatCompletion response."""
        content = "".join(self.generated_chunks)
        prompt_count = len(self.prompt_tokens)
        completion_count = len(self.generated_tokens)
        return {
            "id": self.request_id,
            "object": "chat.completion",
            "created": int(self.created_at),
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": self.finish_reason or "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_count,
                "completion_tokens": completion_count,
                "total_tokens": prompt_count + completion_count,
            },
        }

    def to_chunk_dict(self, delta_content: str, finish_reason: Optional[str] = None) -> Dict[str, Any]:
        """Format as standard OpenAI ChatCompletionChunk response."""
        delta = {"content": delta_content} if delta_content else {}
        return {
            "id": self.request_id,
            "object": "chat.completion.chunk",
            "created": int(self.created_at),
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }


class PipelineScheduler:
    """Interleaved concurrent request scheduler for FrontierSplit pipeline."""

    def __init__(
        self,
        stage0_url: str = "http://localhost:50051",
        num_workers: int = 8,
        total_stages: int = 4,
        tokenizer: Optional[Any] = None,
    ):
        self.stage0_url = stage0_url
        self.num_workers = num_workers
        self.total_stages = total_stages
        self.tokenizer = tokenizer

        self.active_requests: Dict[str, ScheduledRequest] = {}
        self.ready_queue: asyncio.Queue[ScheduledRequest] = asyncio.Queue()
        self.is_running = False
        self._worker_tasks: List[asyncio.Task] = []

        # Telemetry & Performance Counters
        self.start_time = time.time()
        self.total_submitted: int = 0
        self.total_completed: int = 0
        self.total_tokens_generated: int = 0
        self.step_latencies: List[float] = []

    async def start(self) -> None:
        """Start scheduler worker loop tasks."""
        if self.is_running:
            return
        self.is_running = True
        self.start_time = time.time()
        self._worker_tasks = [
            asyncio.create_task(self._worker_loop(i))
            for i in range(self.num_workers)
        ]

    async def stop(self) -> None:
        """Gracefully stop scheduler worker tasks."""
        if not self.is_running:
            return
        self.is_running = False
        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()

    async def ensure_started(self) -> None:
        """Ensure the background worker pool is active."""
        if not self.is_running:
            await self.start()

    async def submit_request(
        self,
        model: str,
        messages: List[Any],
        max_tokens: int = 64,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> ScheduledRequest:
        """Register a new request and enqueue its initial prefill step."""
        await self.ensure_started()

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        prompt_text = "\n".join([f"{getattr(m, 'role', 'user')}: {getattr(m, 'content', str(m))}" for m in messages])
        
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "apply_chat_template") and messages:
                try:
                    formatted = [
                        {"role": getattr(m, "role", "user"), "content": getattr(m, "content", str(m))}
                        for m in messages
                    ]
                    prompt_tokens = self.tokenizer.apply_chat_template(
                        formatted, add_generation_prompt=True, tokenize=True
                    )
                except Exception:
                    prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)
            else:
                prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        else:
            prompt_tokens = [ord(c) % 32000 for c in prompt_text] if prompt_text else [1]

        req = ScheduledRequest(
            request_id=request_id,
            model=model,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

        self.active_requests[request_id] = req
        self.total_submitted += 1

        # Enqueue prefill step
        await self.ready_queue.put(req)
        return req

    async def _worker_loop(self, worker_id: int) -> None:
        """Continuous pipeline step dispatcher worker."""
        while self.is_running:
            try:
                req = await self.ready_queue.get()
            except asyncio.CancelledError:
                break

            try:
                await self._execute_step(req)
            except Exception as e:
                req.error = str(e)
                req.is_finished = True
                if req.stream:
                    await req.stream_queue.put(None)
                req.done_event.set()
                self._finalize_request(req)
            finally:
                self.ready_queue.task_done()

    async def _execute_step(self, req: ScheduledRequest) -> None:
        """Execute a single pipeline forward pass for a request."""
        packet = ActivationPacket(
            request_id=req.request_id,
            sequence_step=req.current_step,
            stage_id=0,
            is_prefill=req.is_prefill,
            tokens=req.input_tokens,
        )

        # Offload blocking HTTP call to Stage 0 so async loop remains responsive
        def _post_forward() -> Dict[str, Any]:
            resp = requests.post(f"{self.stage0_url}/forward", json=packet.model_dump(), timeout=30)
            resp.raise_for_status()
            return resp.json()

        step_start = time.time()
        result_dict = await asyncio.to_thread(_post_forward)
        gen_result = GenerationResponse.model_validate(result_dict)
        step_latency = (time.time() - step_start) * 1000

        # Update request state
        req.generated_tokens.append(gen_result.token_id)
        req.generated_chunks.append(gen_result.text)
        req.step_latencies.append(step_latency)
        req.stage_timings.append(gen_result.stage_timings)

        # Update telemetry
        self.total_tokens_generated += 1
        self.step_latencies.append(step_latency)
        if len(self.step_latencies) > 200:
            self.step_latencies.pop(0)

        # If streaming, emit the chunk
        if req.stream:
            chunk = req.to_chunk_dict(gen_result.text, finish_reason=None)
            await req.stream_queue.put(chunk)

        # Check termination condition
        reached_max = (req.current_step >= req.max_tokens - 1)
        if gen_result.is_finished or reached_max:
            req.is_finished = True
            req.finish_reason = "stop" if gen_result.is_finished else "length"
            if req.stream:
                finish_chunk = req.to_chunk_dict("", finish_reason=req.finish_reason)
                await req.stream_queue.put(finish_chunk)
                await req.stream_queue.put(None)  # Sentinel to close stream
            req.done_event.set()
            self._finalize_request(req)
        else:
            # Advance step and re-enqueue for next autoregressive decode step
            req.current_step += 1
            req.is_prefill = False
            req.input_tokens = [gen_result.token_id]
            await self.ready_queue.put(req)

    def _finalize_request(self, req: ScheduledRequest) -> None:
        """Retire a finished request and update metrics."""
        self.active_requests.pop(req.request_id, None)
        self.total_completed += 1

    def get_telemetry(self) -> Dict[str, Any]:
        """Compute real-time pipeline performance and bubble metrics."""
        active_count = len(self.active_requests)
        k = self.total_stages
        m = max(1, active_count)

        # Theoretical pipeline bubble fraction: (K - 1) / (M + K - 1)
        bubble_fraction = (k - 1) / (m + k - 1)
        bubble_elimination_pct = (1.0 - bubble_fraction) * 100.0

        elapsed_s = max(0.001, time.time() - self.start_time)
        throughput_tok_per_sec = self.total_tokens_generated / elapsed_s
        avg_step_ms = sum(self.step_latencies) / len(self.step_latencies) if self.step_latencies else 0.0

        return {
            "active_streams": active_count,
            "total_stages": k,
            "total_submitted": self.total_submitted,
            "total_completed": self.total_completed,
            "total_tokens_generated": self.total_tokens_generated,
            "throughput_tokens_per_sec": round(throughput_tok_per_sec, 2),
            "avg_step_latency_ms": round(avg_step_ms, 2),
            "theoretical_bubble_fraction": round(bubble_fraction, 4),
            "bubble_elimination_pct": round(bubble_elimination_pct, 2),
        }
