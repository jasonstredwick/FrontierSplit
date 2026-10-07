"""FrontierSplit Ingress Gateway.

Exposes an OpenAI-compatible API (/v1/chat/completions, /v1/models, /v1/telemetry) to clients
and multi-agent harnesses (DSPy, OpenHands, LangChain).
Orchestrates autoregressive generation across the pipeline cluster using an asynchronous
interleaved request scheduler with streaming support.
"""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import json
import time
from typing import Any, Dict, List, Optional
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from frontiersplit.scheduler import PipelineScheduler, ScheduledRequest


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "frontiersplit-mixtral-8x7b"
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False



class CompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def create_gateway_app(
    stage0_url: str = "http://localhost:50051",
    model_name: str = "frontiersplit-mixtral-8x7b",
    num_workers: int = 8,
    total_stages: int = 4,
    scheduler: Optional[PipelineScheduler] = None,
) -> FastAPI:
    """Create FastAPI application with interleaved pipeline scheduler."""
    if scheduler is None:
        tokenizer = None
        if HAS_TRANSFORMERS and model_name and not model_name.startswith("frontiersplit-test"):
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception:
                tokenizer = None

        scheduler = PipelineScheduler(
            stage0_url=stage0_url,
            num_workers=num_workers,
            total_stages=total_stages,
            tokenizer=tokenizer,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await app.state.scheduler.start()
        yield
        await app.state.scheduler.stop()

    app = FastAPI(title="FrontierSplit OpenAI-Compatible Ingress Gateway", lifespan=lifespan)
    app.state.scheduler = scheduler

    @app.get("/health")
    def health():
        try:
            resp = requests.get(f"{stage0_url}/health", timeout=3)
            return {
                "gateway": "healthy",
                "stage0_status": resp.json(),
                "active_streams": len(scheduler.active_requests),
            }
        except Exception as e:
            return {"gateway": "degraded", "stage0_error": str(e)}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "frontiersplit",
                }
            ],
        }

    @app.get("/v1/telemetry")
    @app.get("/metrics")
    def get_telemetry():
        return scheduler.get_telemetry()

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        req_state = await scheduler.submit_request(
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
            raise HTTPException(status_code=502, detail=f"Pipeline forward error: {req_state.error}")

        return JSONResponse(content=req_state.to_chat_completion_response())

    return app


def main():
    parser = argparse.ArgumentParser(description="FrontierSplit Ingress Gateway")
    parser.add_argument("--stage0-url", type=str, default="http://localhost:50051", help="URL of Stage 0 worker")
    parser.add_argument("--port", type=int, default=8000, help="Gateway port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    parser.add_argument("--model-name", type=str, default="frontiersplit-mixtral-8x7b", help="Model name to advertise")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of concurrent dispatch workers")
    parser.add_argument("--total-stages", type=int, default=4, help="Total pipeline stages in cluster")
    args = parser.parse_args()

    app = create_gateway_app(
        stage0_url=args.stage0_url,
        model_name=args.model_name,
        num_workers=args.num_workers,
        total_stages=args.total_stages,
    )
    print(f"Starting FrontierSplit Gateway on port {args.port}, connected to Stage 0 at {args.stage0_url}...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
