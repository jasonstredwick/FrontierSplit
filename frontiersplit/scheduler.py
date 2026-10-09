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

from frontiersplit.models import resolve_model_spec
from frontiersplit.protocol import (
    ActivationPacket,
    BatchedActivationPacket,
    BatchedGenerationResponse,
    ChunkActivationPacket,
    ChunkGenerationResponse,
    GenerationResponse,
    ReleaseSessionPacket,
    ReleaseSessionResponse,
)
from frontiersplit.transport import BinaryTransportClient, BinaryTransportServer


class ScheduledRequest:
    """Tracks state and history for an active chat completion request."""

    def __init__(
        self,
        request_id: str,
        model: str,
        prompt_text: str,
        prompt_tokens: List[int],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stream: bool = False,
        tokenizer: Optional[Any] = None,
    ):
        self.request_id = request_id
        self.model = model
        self.prompt_text = prompt_text
        self.prompt_tokens = prompt_tokens
        self.tokenizer = tokenizer

        # Align with model specification
        model_spec = resolve_model_spec(model, tokenizer=self.tokenizer)
        if max_tokens is None:
            remaining_ctx = max(1, model_spec.context_window - len(prompt_tokens))
            self.max_tokens = min(model_spec.max_generation_tokens, remaining_ctx)
        else:
            self.max_tokens = min(max_tokens, model_spec.context_window)

        self.temperature = temperature
        self.stream = stream

        self.created_at = time.time()
        self.current_step = 0
        self.is_prefill = True
        self.input_tokens = list(prompt_tokens) if prompt_tokens else [1]

        # Chunked Sequence Processing: S tokens chunked into D blocks of size C=16
        self.chunk_size = 16
        s_len = len(self.input_tokens)
        self.total_prefill_chunks = max(1, (s_len + self.chunk_size - 1) // self.chunk_size)
        self.prefill_chunks: List[List[int]] = []
        self.chunk_valid_lens: List[int] = []
        for d in range(self.total_prefill_chunks):
            start_i = d * self.chunk_size
            end_i = min(start_i + self.chunk_size, s_len)
            sub = self.input_tokens[start_i:end_i]
            v_len = len(sub)
            if v_len < self.chunk_size:
                sub = sub + [0] * (self.chunk_size - v_len)
            self.prefill_chunks.append(sub)
            self.chunk_valid_lens.append(v_len)
        self.current_chunk_idx = 0

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
        if self.tokenizer is not None and self.generated_tokens:
            content = self.tokenizer.decode(self.generated_tokens, skip_special_tokens=True)
        else:
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
        num_workers: int = 1,
        total_stages: int = 4,
        tokenizer: Optional[Any] = None,
        max_batch_size: int = 16,
        use_kv_cache: bool = True,
        stage0_tcp: Optional[str] = None,
        enable_1f1b: bool = True,
        gateway_host: Optional[str] = None,
        reply_port: int = 50060,
        max_in_flight: Optional[int] = None,
    ):
        self.stage0_url = stage0_url
        self.num_workers = num_workers
        self.total_stages = total_stages
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.use_kv_cache = use_kv_cache

        target_peer = stage0_tcp or (
            stage0_url.replace("tcp://", "") if stage0_url and stage0_url.startswith("tcp://") else None
        )
        self.stage0_client = BinaryTransportClient(target_peer) if target_peer else None

        self.enable_1f1b = enable_1f1b and (self.stage0_client is not None)
        self.gateway_host = gateway_host or "127.0.0.1"
        self.reply_port = reply_port
        self.reply_server: Optional[BinaryTransportServer] = None
        self._in_flight_times: Dict[str, float] = {}

        flight_limit = max_in_flight or max(self.total_stages * 4, 32)
        self._in_flight_sem: Optional[asyncio.Semaphore] = asyncio.Semaphore(flight_limit) if self.enable_1f1b else None

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
        """Start scheduler worker loop tasks and optional 1F1B reply server."""
        if self.is_running:
            return
        self.is_running = True
        self.start_time = time.time()

        if self.enable_1f1b:
            try:
                self.reply_server = BinaryTransportServer(
                    host="0.0.0.0",
                    port=self.reply_port,
                    response_handler=self._handle_pipeline_reply,
                )
                await self.reply_server.start()
            except OSError:
                self.reply_server = BinaryTransportServer(
                    host="0.0.0.0",
                    port=0,
                    response_handler=self._handle_pipeline_reply,
                )
                await self.reply_server.start()
            self.reply_port = self.reply_server.port

        self._worker_tasks = [
            asyncio.create_task(self._worker_loop(i))
            for i in range(self.num_workers)
        ]

    async def stop(self) -> None:
        """Gracefully stop scheduler worker tasks, 1F1B reply server, and close binary transport client."""
        if not self.is_running:
            return
        self.is_running = False
        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()

        if self.reply_server is not None:
            await self.reply_server.stop()
            self.reply_server = None

        if self.stage0_client is not None:
            await self.stage0_client.close()

    async def ensure_started(self) -> None:
        """Ensure the background worker pool is active."""
        if not self.is_running:
            await self.start()

    async def submit_request(
        self,
        model: str,
        messages: List[Any],
        max_tokens: Optional[int] = None,
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
                    encoded = self.tokenizer.apply_chat_template(
                        formatted, add_generation_prompt=True, tokenize=True
                    )
                    if isinstance(encoded, dict) or hasattr(encoded, "get"):
                        prompt_tokens = list(encoded.get("input_ids", encoded))
                    elif isinstance(encoded, (list, tuple)):
                        prompt_tokens = list(encoded)
                    elif hasattr(encoded, "tolist"):
                        prompt_tokens = encoded.tolist()
                    else:
                        prompt_tokens = list(encoded)
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
            tokenizer=self.tokenizer,
        )

        self.active_requests[request_id] = req
        self.total_submitted += 1

        # Enqueue prefill step
        await self.ready_queue.put(req)
        return req

    async def _worker_loop(self, worker_id: int) -> None:
        """Continuous pipeline step dispatcher worker for chunked sequence execution (B removed)."""
        while self.is_running:
            try:
                req = await self.ready_queue.get()
            except asyncio.CancelledError:
                break

            try:
                await self._execute_chunk_step(req)
            except Exception as e:
                req.error = str(e)
                req.is_finished = True
                if req.stream:
                    await req.stream_queue.put(None)
                req.done_event.set()
                self._finalize_request(req)
            finally:
                self.ready_queue.task_done()

    async def _execute_chunk_step(self, req: ScheduledRequest) -> None:
        """Execute a single atomic chunk step of fixed size C=16 for the given request.
        Tensor shape is strictly [16, 4096] with ZERO variable batch dimension B.
        """
        if req.is_prefill:
            chunk_idx = req.current_chunk_idx
            tokens = req.prefill_chunks[chunk_idx]
            valid_tokens = req.chunk_valid_lens[chunk_idx]
            total_chunks = req.total_prefill_chunks
        else:
            chunk_idx = req.total_prefill_chunks + len(req.generated_tokens) - 1
            tokens = [req.generated_tokens[-1]] + [0] * 15
            valid_tokens = 1
            total_chunks = chunk_idx + 1

        packet = ChunkActivationPacket(
            request_id=req.request_id,
            chunk_idx=chunk_idx,
            total_chunks=total_chunks,
            chunk_size=16,
            valid_tokens=valid_tokens,
            is_prefill=req.is_prefill,
            stage_id=0,
            tokens=tokens,
            tensor_shape=[16, 4096],
            tensor_dtype="float16",
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )

        if self.enable_1f1b and self.stage0_client is not None:
            packet.reply_to = f"{self.gateway_host}:{self.reply_port}"
            self._in_flight_times[req.request_id] = time.time()
            if self._in_flight_sem is not None:
                await self._in_flight_sem.acquire()
            try:
                await self.stage0_client.send_async_forward(packet)
            except Exception as e:
                if self._in_flight_sem is not None:
                    self._in_flight_sem.release()
                raise e
            return

        step_start = time.time()
        def _post_chunk() -> Dict[str, Any]:
            resp = requests.post(f"{self.stage0_url}/forward_chunk", json=packet.model_dump(), timeout=600)
            resp.raise_for_status()
            return resp.json()

        result_dict = await asyncio.to_thread(_post_chunk)
        chunk_result = ChunkGenerationResponse.model_validate(result_dict)
        step_latency = (time.time() - step_start) * 1000
        await self._handle_chunk_reply(chunk_result, step_latency=step_latency)

    async def _handle_chunk_reply(self, resp: ChunkGenerationResponse, step_latency: Optional[float] = None) -> None:
        """Processes ChunkGenerationResponse from final stage."""
        req = self.active_requests.get(resp.request_id)
        if not req:
            return

        now = time.time()
        dispatch_time = self._in_flight_times.pop(req.request_id, None)
        computed_latency = (now - dispatch_time) * 1000 if dispatch_time else (step_latency or 0.0)

        self.step_latencies.append(computed_latency)
        if len(self.step_latencies) > 200:
            self.step_latencies.pop(0)

        if req.is_prefill:
            req.current_chunk_idx += 1
            if req.current_chunk_idx < req.total_prefill_chunks:
                # Dispatch next prefill chunk
                await self.ready_queue.put(req)
                return
            else:
                # Prefill completed!
                req.is_prefill = False

        if resp.next_token_id is not None:
            token_id = resp.next_token_id
            req.generated_tokens.append(token_id)
            self.total_tokens_generated += 1

            token_text = (
                self.tokenizer.decode([token_id], skip_special_tokens=True)
                if self.tokenizer is not None
                else f" tok{token_id}"
            )
            req.generated_chunks.append(token_text)
            req.step_latencies.append(computed_latency)
            req.stage_timings.append(resp.stage_timings)

            if req.stream:
                chunk_dict = req.to_chunk_dict(token_text, finish_reason=None)
                await req.stream_queue.put(chunk_dict)

            reached_max = (req.max_tokens is not None and len(req.generated_tokens) >= req.max_tokens)
            if resp.is_finished or reached_max:
                req.is_finished = True
                req.finish_reason = resp.finish_reason or ("stop" if resp.is_finished else "length")
                if req.stream:
                    finish_chunk = req.to_chunk_dict("", finish_reason=req.finish_reason)
                    await req.stream_queue.put(finish_chunk)
                    await req.stream_queue.put(None)
                req.done_event.set()
                self._finalize_request(req)
            else:
                # Queue next decode chunk
                await self.ready_queue.put(req)
        else:
            if resp.is_finished:
                req.is_finished = True
                req.finish_reason = resp.finish_reason or "stop"
                if req.stream:
                    await req.stream_queue.put(None)
                req.done_event.set()
                self._finalize_request(req)

    async def _handle_pipeline_reply(self, resp: Union[BatchedGenerationResponse, ChunkGenerationResponse]) -> None:
        """Processes responses received directly from the final stage over persistent TCP in 1F1B mode."""
        if self._in_flight_sem is not None:
            self._in_flight_sem.release()

        if isinstance(resp, ChunkGenerationResponse):
            await self._handle_chunk_reply(resp)
            return

        now = time.time()
        for gen_result in resp.responses:
            req = self.active_requests.get(gen_result.request_id)
            if not req:
                continue

            dispatch_time = self._in_flight_times.pop(req.request_id, None)
            step_latency = (now - dispatch_time) * 1000 if dispatch_time else gen_result.latency_ms

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
            reached_max = (req.max_tokens is not None and req.current_step >= req.max_tokens - 1)
            if gen_result.is_finished or reached_max:
                req.is_finished = True
                req.finish_reason = "stop" if gen_result.is_finished else "length"
                if req.stream:
                    finish_chunk = req.to_chunk_dict("", finish_reason=req.finish_reason)
                    await req.stream_queue.put(finish_chunk)
                    await req.stream_queue.put(None)
                req.done_event.set()
                self._finalize_request(req)
            else:
                req.current_step += 1
                req.is_prefill = False
                req.input_tokens = req.prompt_tokens + req.generated_tokens
                await self.ready_queue.put(req)

    async def _execute_step(self, req: ScheduledRequest) -> None:
        """Execute a single pipeline forward pass for a request."""
        if self.stage0_client is not None:
            await self._execute_batched_step([req])
            return

        if self.use_kv_cache and not req.is_prefill:
            step_tokens = [req.generated_tokens[-1]]
        else:
            step_tokens = req.input_tokens

        packet = ActivationPacket(
            request_id=req.request_id,
            sequence_step=req.current_step,
            stage_id=0,
            is_prefill=req.is_prefill,
            use_kv_cache=self.use_kv_cache,
            max_tokens=req.max_tokens,
            tokens=step_tokens,
        )

        # Offload blocking HTTP call to Stage 0 so async loop remains responsive
        def _post_forward() -> Dict[str, Any]:
            resp = requests.post(f"{self.stage0_url}/forward", json=packet.model_dump(), timeout=600)
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
            req.input_tokens = req.prompt_tokens + req.generated_tokens
            await self.ready_queue.put(req)

    def _finalize_request(self, req: ScheduledRequest) -> None:
        """Retire a finished request, reclaim worker KV cache, and update metrics."""
        self.active_requests.pop(req.request_id, None)
        self.total_completed += 1
        if self.use_kv_cache:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._async_release_session(req.request_id))
            except RuntimeError:
                pass

    async def _async_release_session(self, request_id: str) -> None:
        """Asynchronously notify stage 0 to release worker session KV cache."""
        if self.stage0_client is not None:
            try:
                await self.stage0_client.send_release_sessions([request_id])
            except Exception:
                pass
        else:
            def _post_release():
                try:
                    packet = ReleaseSessionPacket(request_ids=[request_id])
                    requests.post(f"{self.stage0_url}/release_sessions", json=packet.model_dump(), timeout=10)
                except Exception:
                    pass
            await asyncio.to_thread(_post_release)

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
