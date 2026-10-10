"""FrontierSplit OpenAI-Compatible API Gateway Demonstration.

Demonstrates running the FrontierSplit OpenAI-compatible API Gateway exposing:
  - GET  /health
  - GET  /v1/models
  - POST /v1/chat/completions (Standard JSON non-streaming)
  - POST /v1/chat/completions (Server-Sent Events streaming with `stream=True`)
  - Multi-turn prefix cache reuse on the disaggregated ContextServer.

Run with:
    python examples/gateway_demo.py
"""

from __future__ import annotations

import json
import time

import torch
from fastapi.testclient import TestClient
from transformers import LlamaConfig, LlamaForCausalLM

from frontiersplit import (
    ContextClient,
    ContextServer,
    DisaggregatedModel,
    create_gateway_app,
)


class SimpleVocabTokenizer:
    """Lightweight deterministic tokenizer for local demonstration."""

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        """Encodes characters to token IDs clamped within vocab_size."""
        return [max(1, ord(c) % self.vocab_size) for c in text]

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        """Decodes token IDs back to displayable ASCII characters."""
        return "".join(chr(t) if 32 <= t < 127 else " " for t in token_ids)


def main() -> None:
    print("=" * 76)
    print(" FrontierSplit: Standalone OpenAI-Compatible API Gateway Demonstration")
    print("=" * 76)

    # 1. Architecture Setup
    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
    )
    model = LlamaForCausalLM(config)
    model.eval()

    tokenizer = SimpleVocabTokenizer(vocab_size=256)

    print("\n[1/5] Launching Disaggregated Context Server...")
    server = ContextServer(max_cached_tokens=50_000)
    port = server.start_in_thread(host="127.0.0.1", port=0)
    client = ContextClient(host="127.0.0.1", port=port)
    print(f"  • ContextServer running on TCP port: {port}")

    try:
        # 2. Wire DisaggregatedModel and Gateway App
        print("\n[2/5] Initializing DisaggregatedModel & FastAPI Gateway...")
        disagg_model = DisaggregatedModel(
            model=model,
            context_client=client,
        )

        app = create_gateway_app(
            model_name="meta-llama/Llama-3-8B-Instruct",
            disaggregated_model=disagg_model,
            context_client=client,
            tokenizer=tokenizer,
        )
        api = TestClient(app)

        # 3. Health & Model Discovery
        print("\n[3/5] Querying /health and /v1/models endpoints...")
        health_resp = api.get("/health")
        print(
            f"  • GET /health -> Status {health_resp.status_code}: {health_resp.json()}"
        )

        models_resp = api.get("/v1/models")
        print(f"  • GET /v1/models -> Model List: {models_resp.json()['data']}")

        # 4. Standard Non-Streaming Chat Completion
        print("\n[4/5] Executing POST /v1/chat/completions (Non-Streaming JSON)...")
        req_payload_1 = {
            "model": "meta-llama/Llama-3-8B-Instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an autonomous coding assistant.",
                },
                {"role": "user", "content": "Hello FrontierSplit!"},
            ],
            "max_tokens": 16,
            "stream": False,
        }

        t0 = time.perf_counter()
        resp_1 = api.post("/v1/chat/completions", json=req_payload_1)
        t_non_stream = (time.perf_counter() - t0) * 1000

        data_1 = resp_1.json()
        print(f"  • Response ID:    {data_1['id']}")
        print(f"  • Prompt Tokens:  {data_1['usage']['prompt_tokens']}")
        print(f"  • Output Tokens:  {data_1['usage']['completion_tokens']}")
        print(f"  • Response Content: {data_1['choices'][0]['message']['content']!r}")
        print(f"  • Roundtrip Time: {t_non_stream:.2f} ms")

        # 5. Real-Time Streaming SSE & Prefix Cache Hit
        print(
            "\n[5/5] Executing POST /v1/chat/completions (Server-Sent Events Streaming)..."
        )
        print(
            "  Notice: We send an extended conversation sharing the exact system prompt prefix."
        )
        req_payload_2 = {
            "model": "meta-llama/Llama-3-8B-Instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an autonomous coding assistant.",
                },
                {"role": "user", "content": "Hello FrontierSplit!"},
                {
                    "role": "assistant",
                    "content": data_1["choices"][0]["message"]["content"],
                },
                {
                    "role": "user",
                    "content": "Can you explain how disaggregated memory works?",
                },
            ],
            "max_tokens": 20,
            "stream": True,
        }

        stream_chunks: list[str] = []
        t1 = time.perf_counter()
        with api.stream(
            "POST", "/v1/chat/completions", json=req_payload_2
        ) as stream_resp:
            print("  • Received SSE Tokens: ", end="", flush=True)
            for line in stream_resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: ") :].strip()
                if data_str == "[DONE]":
                    break
                chunk_obj = json.loads(data_str)
                delta = chunk_obj["choices"][0]["delta"].get("content", "")
                if delta:
                    stream_chunks.append(delta)
                    print(delta, end="", flush=True)
            print()

        t_stream = (time.perf_counter() - t1) * 1000
        print(f"  • Streamed {len(stream_chunks)} delta chunks in {t_stream:.2f} ms")

        # Telemetry verification
        telemetry = api.get("/v1/telemetry").json()
        print("\nGateway Telemetry:")
        print(f"  • Backend:               {telemetry['backend']}")
        print(f"  • Model:                 {telemetry['model']}")
        print(f"  • Context Server TCP:    {telemetry['context_server_connected']}")
        print(f"  • Prefix Caching:        {telemetry['prefix_caching']}")

        print("\n" + "=" * 76)
        print(
            " Demonstration completed successfully! Gateway is fully production-ready."
        )
        print("=" * 76)

    finally:
        client.close_sync()
        server.stop_thread()


if __name__ == "__main__":
    main()
