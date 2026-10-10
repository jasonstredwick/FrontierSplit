"""Integration tests for FrontierSplit OpenAI-Compatible Ingress Gateway.

Validates /v1/chat/completions (both non-streaming JSON and streaming SSE),
/v1/models, /health, and prefix cache hit handling over DisaggregatedModel.
"""

from __future__ import annotations

import json

import pytest
import torch
from fastapi.testclient import TestClient
from transformers import LlamaConfig, LlamaForCausalLM

from frontiersplit import (
    ContextClient,
    ContextServer,
    DisaggregatedModel,
    create_gateway_app,
)


@pytest.fixture
def disagg_cluster():
    """Fixture providing a tiny Llama model and ContextServer cluster."""
    torch.manual_seed(42)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()

    server = ContextServer()
    port = server.start_in_thread(host="127.0.0.1", port=0)
    client = ContextClient(host="127.0.0.1", port=port)
    disagg_model = DisaggregatedModel(model=model, context_client=client)

    app = create_gateway_app(
        model_name="test-disagg-llama",
        disaggregated_model=disagg_model,
        context_client=client,
    )
    test_client = TestClient(app)

    try:
        yield test_client, client, server
    finally:
        client.close_sync()
        server.stop_thread()


def test_gateway_health_and_models(disagg_cluster):
    """Verify /health, /v1/models, and /v1/telemetry endpoints."""
    test_client, _, _ = disagg_cluster

    # 1. Health check
    resp = test_client.get("/health")
    assert resp.status_code == 200
    health_data = resp.json()
    assert health_data["gateway"] == "healthy"
    assert health_data["backend"] == "disaggregated"

    # 2. List models
    resp_models = test_client.get("/v1/models")
    assert resp_models.status_code == 200
    models_data = resp_models.json()
    assert models_data["object"] == "list"
    assert len(models_data["data"]) == 1
    assert models_data["data"][0]["id"] == "test-disagg-llama"

    # 3. Telemetry
    resp_telem = test_client.get("/v1/telemetry")
    assert resp_telem.status_code == 200
    assert resp_telem.json()["backend"] == "disaggregated"


def test_gateway_disaggregated_non_streaming(disagg_cluster):
    """Verify non-streaming /v1/chat/completions returns OpenAI-compliant JSON."""
    test_client, _, _ = disagg_cluster

    req_body = {
        "model": "test-disagg-llama",
        "messages": [
            {"role": "user", "content": "10 20 30 40"},
        ],
        "max_tokens": 5,
        "stream": False,
    }

    resp = test_client.post("/v1/chat/completions", json=req_body)
    assert resp.status_code == 200

    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "test-disagg-llama"
    assert len(data["choices"]) == 1

    choice = data["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert len(choice["message"]["content"]) > 0

    usage = data["usage"]
    assert usage["prompt_tokens"] == 4
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 9


def test_gateway_disaggregated_streaming_sse(disagg_cluster):
    """Verify streaming /v1/chat/completions yields Server-Sent Events (SSE)."""
    test_client, _, _ = disagg_cluster

    req_body = {
        "model": "test-disagg-llama",
        "messages": [
            {"role": "user", "content": "5 15 25 35 45"},
        ],
        "max_tokens": 4,
        "stream": True,
    }

    resp = test_client.post("/v1/chat/completions", json=req_body)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = [line.strip() for line in resp.text.split("\n") if line.strip()]

    # Verify SSE prefix format
    for line in lines:
        assert line.startswith("data: ")

    # Last line must be [DONE]
    assert lines[-1] == "data: [DONE]"

    # Parse JSON payloads
    chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]

    assert len(chunks) >= 4
    # First chunk introduces role
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    # Last chunk has finish_reason stop
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_gateway_disaggregated_prefix_cache_reuse(disagg_cluster):
    """Verify multiple chat requests reuse cached prompt prefixes through gateway."""
    test_client, client, _ = disagg_cluster

    shared_prompt = "12 24 36 48 60 72 84 96"

    # Request 1: Cold start
    req_1 = {
        "model": "test-disagg-llama",
        "messages": [{"role": "user", "content": shared_prompt}],
        "max_tokens": 4,
        "stream": False,
    }
    resp_1 = test_client.post("/v1/chat/completions", json=req_1)
    assert resp_1.status_code == 200
    tokens_1 = resp_1.json()["choices"][0]["message"]["content"]

    # Verify ContextServer indexed the prompt
    token_list = [12, 24, 36, 48, 60, 72, 84, 96]
    matched_len, _ = client.match_prefix_sync(token_list)
    assert matched_len == len(token_list)

    # Request 2: Warm hit (bypasses prefill)
    req_2 = {
        "model": "test-disagg-llama",
        "messages": [{"role": "user", "content": shared_prompt}],
        "max_tokens": 4,
        "stream": False,
    }
    resp_2 = test_client.post("/v1/chat/completions", json=req_2)
    assert resp_2.status_code == 200
    tokens_2 = resp_2.json()["choices"][0]["message"]["content"]

    # Both requests must produce bit-for-bit identical content
    assert tokens_1 == tokens_2
