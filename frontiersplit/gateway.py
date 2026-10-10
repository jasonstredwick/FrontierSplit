"""FrontierSplit OpenAI-Compatible Ingress Gateway.

Exposes an OpenAI-compatible REST API (/v1/chat/completions, /v1/models, /health)
supporting both streaming (Server-Sent Events) and non-streaming inference.
Supports both disaggregated attention backends (DisaggregatedModel + ContextServer)
and distributed pipeline schedulers.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from frontiersplit.context_server import ContextClient
from frontiersplit.hf_model import DisaggregatedModel
from frontiersplit.scheduler import PipelineScheduler

try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class ChatMessage(BaseModel):
    """A single chat message in a completion request/response."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request schema."""

    model: str = "frontiersplit-model"
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False
    use_prefix_cache: bool = True


class CompletionChoice(BaseModel):
    """A single choice in a non-streaming chat completion response."""

    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    """Token usage metrics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response schema."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


def create_gateway_app(
    stage0_url: str = "http://localhost:50051",
    model_name: str = "frontiersplit-model",
    num_workers: int = 1,
    total_stages: int = 4,
    scheduler: PipelineScheduler | None = None,
    max_batch_size: int = 16,
    use_kv_cache: bool = True,
    stage0_tcp: str | None = None,
    enable_1f1b: bool = True,
    gateway_host: str | None = None,
    reply_port: int = 50060,
    max_in_flight: int | None = None,
    disaggregated_model: DisaggregatedModel | None = None,
    context_client: ContextClient | None = None,
    tokenizer: Any | None = None,
) -> FastAPI:
    """Creates a FastAPI gateway application.

    Args:
        stage0_url: HTTP endpoint of Stage 0 worker (for pipeline scheduler mode).
        model_name: Model identifier advertised in /v1/models and completions.
        num_workers: Concurrent dispatch workers for pipeline scheduler.
        total_stages: Total stages in pipeline mode.
        scheduler: Optional existing PipelineScheduler instance.
        max_batch_size: Maximum batch size for scheduler queue.
        use_kv_cache: Whether KV caching is enabled for pipeline scheduler.
        stage0_tcp: Optional persistent binary TCP endpoint for Stage 0.
        enable_1f1b: Whether 1F1B pipelining is enabled.
        gateway_host: Host for downstream stage direct streaming.
        reply_port: Binary TCP reply port for direct response streaming.
        max_in_flight: Limit on concurrent in-flight micro-batches.
        disaggregated_model: Optional DisaggregatedModel instance for direct serving.
        context_client: Optional ContextClient instance for prefix cache queries.
        tokenizer: Optional tokenizer instance for text encoding/decoding.

    Returns:
        Configured FastAPI application instance.
    """
    if disaggregated_model is None and scheduler is None:
        tok = None
        if (
            HAS_TRANSFORMERS
            and model_name
            and not model_name.startswith("frontiersplit-test")
        ):
            try:
                tok = AutoTokenizer.from_pretrained(model_name)
            except Exception:
                tok = None

        scheduler = PipelineScheduler(
            stage0_url=stage0_url,
            num_workers=num_workers,
            total_stages=total_stages,
            tokenizer=tok,
            max_batch_size=max_batch_size,
            use_kv_cache=use_kv_cache,
            stage0_tcp=stage0_tcp,
            enable_1f1b=enable_1f1b,
            gateway_host=gateway_host,
            reply_port=reply_port,
            max_in_flight=max_in_flight,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.scheduler is not None:
            await app.state.scheduler.start()
        yield
        if app.state.scheduler is not None:
            await app.state.scheduler.stop()

    app = FastAPI(
        title="FrontierSplit OpenAI-Compatible Ingress Gateway",
        lifespan=lifespan,
    )
    app.state.scheduler = scheduler
    app.state.disaggregated_model = disaggregated_model
    app.state.context_client = context_client
    app.state.tokenizer = tokenizer
    app.state.model_name = model_name

    @app.get("/health")
    def health():
        """Healthcheck endpoint reporting gateway status and backend connectivity."""
        if app.state.disaggregated_model is not None:
            return {
                "gateway": "healthy",
                "backend": "disaggregated",
                "model": app.state.model_name,
            }
        try:
            resp = requests.get(f"{stage0_url}/health", timeout=3)
            active_count = (
                len(app.state.scheduler.active_requests) if app.state.scheduler else 0
            )
            return {
                "gateway": "healthy",
                "stage0_status": resp.json(),
                "active_streams": active_count,
            }
        except Exception as e:
            return {"gateway": "degraded", "stage0_error": str(e)}

    @app.get("/v1/models")
    def list_models():
        """Lists available models matching the OpenAI /v1/models endpoint."""
        return {
            "object": "list",
            "data": [
                {
                    "id": app.state.model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "frontiersplit",
                }
            ],
        }

    @app.get("/v1/telemetry")
    @app.get("/metrics")
    def get_telemetry():
        """Telemetry and metrics endpoint."""
        if app.state.scheduler is not None:
            return app.state.scheduler.get_telemetry()
        return {
            "backend": "disaggregated",
            "model": app.state.model_name,
            "context_server_connected": app.state.context_client is not None,
            "prefix_caching": app.state.context_client is not None,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        """OpenAI-compatible chat completion endpoint."""
        # 1. Disaggregated Model Execution Path
        if app.state.disaggregated_model is not None:
            disagg = app.state.disaggregated_model
            tok = app.state.tokenizer
            active_model = req.model or app.state.model_name

            # Format and tokenize prompt
            if tok is not None:
                if hasattr(tok, "apply_chat_template"):
                    try:
                        msgs = [
                            {"role": m.role, "content": m.content} for m in req.messages
                        ]
                        token_ids = tok.apply_chat_template(
                            msgs, add_generation_prompt=True, tokenize=True
                        )
                    except Exception:
                        text = (
                            "\n".join(f"{m.role}: {m.content}" for m in req.messages)
                            + "\nassistant:"
                        )
                        token_ids = tok.encode(text)
                else:
                    text = (
                        "\n".join(f"{m.role}: {m.content}" for m in req.messages)
                        + "\nassistant:"
                    )
                    token_ids = tok.encode(text)
            else:
                raw_text = req.messages[-1].content
                try:
                    token_ids = [
                        int(x) for x in raw_text.replace(",", " ").split() if x.strip()
                    ]
                    if not token_ids:
                        token_ids = [ord(c) % 128 for c in raw_text] or [1]
                except ValueError:
                    token_ids = [ord(c) % 128 for c in raw_text] or [1]

            input_tensor = torch.tensor([token_ids], dtype=torch.long)
            req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            session_id = f"sess-{req_id}"
            created_time = int(time.time())
            max_new_tokens = req.max_tokens if req.max_tokens is not None else 32

            if req.stream:

                async def event_generator():
                    # Initial role delta chunk
                    role_chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": active_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": ""},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(role_chunk)}\n\n"

                    async for token_id in disagg.generate_stream_async(
                        input_ids=input_tensor,
                        max_new_tokens=max_new_tokens,
                        session_id=session_id,
                        use_prefix_cache=req.use_prefix_cache,
                    ):
                        delta_text = (
                            tok.decode([token_id])
                            if tok is not None
                            else str(token_id) + " "
                        )
                        chunk = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": active_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": delta_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    # Final finish reason chunk
                    final_chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": active_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    event_generator(), media_type="text/event-stream"
                )

            # Non-streaming mode
            generated_token_ids = await disagg.generate_async(
                input_ids=input_tensor,
                max_new_tokens=max_new_tokens,
                session_id=session_id,
                use_prefix_cache=req.use_prefix_cache,
            )
            content = (
                tok.decode(generated_token_ids)
                if tok is not None
                else " ".join(str(t) for t in generated_token_ids)
            )
            prompt_count = len(token_ids)
            comp_count = len(generated_token_ids)

            response_payload = {
                "id": req_id,
                "object": "chat.completion",
                "created": created_time,
                "model": active_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_count,
                    "completion_tokens": comp_count,
                    "total_tokens": prompt_count + comp_count,
                },
            }
            return JSONResponse(content=response_payload)

        # 2. Pipeline Scheduler Execution Path
        assert app.state.scheduler is not None
        req_state = await app.state.scheduler.submit_request(
            model=req.model,
            messages=req.messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stream=req.stream,
        )

        if req.stream:

            async def event_generator():
                while True:
                    chunk = await req_state.stream_queue.get()
                    if chunk is None:
                        yield "data: [DONE]\n\n"
                        break
                    yield f"data: {json.dumps(chunk)}\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # Non-streaming mode: wait for generation to complete
        await req_state.done_event.wait()
        if req_state.error:
            raise HTTPException(
                status_code=502,
                detail=f"Pipeline forward error: {req_state.error}",
            )

        return JSONResponse(content=req_state.to_chat_completion_response())

    return app


def main():
    """Command-line entry point for FrontierSplit Ingress Gateway."""
    parser = argparse.ArgumentParser(description="FrontierSplit Ingress Gateway")
    parser.add_argument(
        "--stage0-url",
        type=str,
        default="http://localhost:50051",
        help="URL of Stage 0 worker",
    )
    parser.add_argument(
        "--stage0-tcp",
        type=str,
        default=None,
        help="Persistent binary TCP address of Stage 0 worker (host:port)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Gateway port to listen on"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    parser.add_argument(
        "--model-name",
        type=str,
        default="frontiersplit-model",
        help="Model name to advertise",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of concurrent dispatch workers",
    )
    parser.add_argument(
        "--total-stages", type=int, default=4, help="Total pipeline stages in cluster"
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=16,
        help="Maximum batch size for dynamic queue draining",
    )
    parser.add_argument(
        "--disable-kv-cache",
        action="store_true",
        help="Disable stateful KV cache and use stateless recomputation",
    )
    parser.add_argument(
        "--disable-1f1b",
        action="store_true",
        help="Disable 1F1B async pipelining and use synchronous round-trip dispatch",
    )
    parser.add_argument(
        "--gateway-host",
        type=str,
        default=None,
        help="Host address for downstream stages to stream replies back to",
    )
    parser.add_argument(
        "--reply-port",
        type=int,
        default=50060,
        help="Persistent binary TCP reply port for 1F1B direct response streaming",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help="Maximum concurrent in-flight micro-batches across pipeline",
    )
    args = parser.parse_args()

    app = create_gateway_app(
        stage0_url=args.stage0_url,
        stage0_tcp=args.stage0_tcp,
        model_name=args.model_name,
        num_workers=args.num_workers,
        total_stages=args.total_stages,
        max_batch_size=args.max_batch_size,
        use_kv_cache=not args.disable_kv_cache,
        enable_1f1b=not args.disable_1f1b,
        gateway_host=args.gateway_host,
        reply_port=args.reply_port,
        max_in_flight=args.max_in_flight,
    )
    print(f"Starting FrontierSplit Gateway on port {args.port}...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
