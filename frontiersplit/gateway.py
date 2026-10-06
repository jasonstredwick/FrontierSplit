"""FrontierSplit Ingress Gateway.

Exposes an OpenAI-compatible API (/v1/chat/completions, /v1/models) to clients
and multi-agent harnesses (DSPy, OpenHands, LangChain).
Orchestrates autoregressive generation across the pipeline cluster.
"""

from __future__ import annotations

import argparse
import time
import uuid
from typing import Any, Dict, List, Optional
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from frontiersplit.protocol import ActivationPacket, GenerationResponse


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "frontiersplit-mixtral-8x7b"
    messages: List[ChatMessage]
    max_tokens: int = 64
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


def create_gateway_app(stage0_url: str = "http://localhost:50051", model_name: str = "frontiersplit-mixtral-8x7b") -> FastAPI:
    app = FastAPI(title="FrontierSplit OpenAI-Compatible Ingress Gateway")

    @app.get("/health")
    def health():
        try:
            resp = requests.get(f"{stage0_url}/health", timeout=3)
            return {"gateway": "healthy", "stage0_status": resp.json()}
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

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(req: ChatCompletionRequest):
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())

        # Concatenate message contents to simulate tokenization
        prompt_text = "\n".join([f"{m.role}: {m.content}" for m in req.messages])
        # Simple ASCII pseudo-tokenizer for demonstration / testing
        tokens = [ord(c) % 32000 for c in prompt_text] if prompt_text else [1]
        prompt_tokens_count = len(tokens)

        generated_tokens = []
        generated_text_chunks = []

        # 1. Prefill step: Send initial prompt tokens to Stage 0
        packet = ActivationPacket(
            request_id=request_id,
            sequence_step=0,
            stage_id=0,
            is_prefill=True,
            tokens=tokens,
        )

        for step in range(req.max_tokens):
            try:
                resp = requests.post(f"{stage0_url}/forward", json=packet.model_dump(), timeout=30)
                resp.raise_for_status()
                gen_result = GenerationResponse.model_validate(resp.json())
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Pipeline forward error at step {step}: {e}")

            generated_tokens.append(gen_result.token_id)
            generated_text_chunks.append(gen_result.text)

            if gen_result.is_finished:
                break

            # 2. Decode step: Send next token back into Stage 0 for autoregressive cycle
            packet = ActivationPacket(
                request_id=request_id,
                sequence_step=step + 1,
                stage_id=0,
                is_prefill=False,
                tokens=[gen_result.token_id],
            )

        completion_text = "".join(generated_text_chunks)
        completion_tokens_count = len(generated_tokens)

        return ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=req.model,
            choices=[
                CompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=completion_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens_count,
                completion_tokens=completion_tokens_count,
                total_tokens=prompt_tokens_count + completion_tokens_count,
            ),
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="FrontierSplit Ingress Gateway")
    parser.add_argument("--stage0-url", type=str, default="http://localhost:50051", help="URL of Stage 0 worker")
    parser.add_argument("--port", type=int, default=8000, help="Gateway port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    parser.add_argument("--model-name", type=str, default="frontiersplit-mixtral-8x7b", help="Model name to advertise")
    args = parser.parse_args()

    app = create_gateway_app(stage0_url=args.stage0_url, model_name=args.model_name)
    print(f"Starting FrontierSplit Gateway on port {args.port}, connected to Stage 0 at {args.stage0_url}...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
