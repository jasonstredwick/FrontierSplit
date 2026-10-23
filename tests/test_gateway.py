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
    assert choice["finish_reason"] in ("stop", "length")
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
    # Last chunk has finish_reason stop or length
    assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")


def test_gateway_disaggregated_prefix_cache_reuse(disagg_cluster):
    """Verify multiple chat requests reuse cached prompt prefixes through gateway."""
    test_client, client, _ = disagg_cluster

    shared_prompt = "12 24 36 48 60 72 84 96"

    # Request 1: Cold start
    req_1 = {
        "model": "test-disagg-llama",
        "messages": [{"role": "user", "content": shared_prompt}],
        "max_tokens": 4,
        "temperature": 0.0,
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
        "temperature": 0.0,
        "stream": False,
    }
    resp_2 = test_client.post("/v1/chat/completions", json=req_2)
    assert resp_2.status_code == 200
    tokens_2 = resp_2.json()["choices"][0]["message"]["content"]

    # Both requests must produce bit-for-bit identical content
    assert tokens_1 == tokens_2


def test_gateway_streaming_multitoken_utf8_and_sampling():
    """Verify streaming SSE buffers multi-token UTF-8 characters (e.g. '€') without replacement chars."""

    class ByteFallbackTokenizer:
        """Tokenizer mapping token IDs to UTF-8 bytes so multi-byte chars span multiple tokens."""

        def __init__(self, seq_bytes: bytes) -> None:
            self._seq_bytes = seq_bytes

        def encode(self, text: str) -> list[int]:
            return [1, 2, 3, 4]

        def decode(self, token_ids: list[int]) -> str:
            # Map generated token index order onto the multi-byte UTF-8 sequence for "€" (0xE2 0x82 0xAC)
            raw = bytes(self._seq_bytes[: len(token_ids)])
            return raw.decode("utf-8", errors="replace")

    torch.manual_seed(42)
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()

    server = ContextServer()
    port = server.start_in_thread(host="127.0.0.1", port=0)
    client = ContextClient(host="127.0.0.1", port=port)
    disagg_model = DisaggregatedModel(model=model, context_client=client)
    tok = ByteFallbackTokenizer("€".encode())  # 3 bytes: 0xE2, 0x82, 0xAC

    app = create_gateway_app(
        model_name="test-utf8-llama",
        disaggregated_model=disagg_model,
        context_client=client,
        tokenizer=tok,
    )
    test_client = TestClient(app)

    try:
        # 1. Non-streaming returns "€"
        resp_non_stream = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-utf8-llama",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 3,
                "temperature": 0.0,
                "stream": False,
            },
        )
        assert resp_non_stream.status_code == 200
        assert resp_non_stream.json()["choices"][0]["message"]["content"] == "€"

        # 2. Streaming reconstructs "€" without emitting '\ufffd'
        resp_stream = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-utf8-llama",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 3,
                "temperature": 0.0,
                "stream": True,
            },
        )
        assert resp_stream.status_code == 200
        lines = [line.strip() for line in resp_stream.text.split("\n") if line.strip()]
        chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
        streamed_text = "".join(
            c["choices"][0]["delta"].get("content", "") for c in chunks
        )
        assert streamed_text == "€"
        assert "\ufffd" not in streamed_text

        # 3. Invalid sampling parameters are rejected with HTTP 400
        bad_resp = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-utf8-llama",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": -1.0,
            },
        )
        assert bad_resp.status_code == 400
    finally:
        client.close_sync()
        server.stop_thread()


def test_gateway_session_persistence_headers(disagg_cluster):
    """Verify X-FrontierSplit-Session-ID header persists session across conversational turns."""
    test_client, client, server = disagg_cluster
    session_id = "agent-session-persist-42"
    prompt_tokens = "1 2 3 4 5"

    headers = {
        "X-FrontierSplit-Session-ID": session_id,
    }

    # Turn 1: Non-streaming with explicit session ID
    resp1 = test_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-disagg-llama",
            "messages": [{"role": "user", "content": prompt_tokens}],
            "max_tokens": 3,
            "temperature": 0.0,
            "stream": False,
        },
        headers=headers,
    )
    assert resp1.status_code == 200
    assert resp1.headers.get("X-FrontierSplit-Session-ID") == session_id

    # Verify session is retained in ContextStore (release_on_completion was False)
    assert server.store.has_session(session_id)

    # Turn 2: Streaming with same session ID header
    resp2 = test_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-disagg-llama",
            "messages": [{"role": "user", "content": prompt_tokens}],
            "max_tokens": 3,
            "temperature": 0.0,
            "stream": True,
        },
        headers=headers,
    )
    assert resp2.status_code == 200
    assert resp2.headers.get("X-FrontierSplit-Session-ID") == session_id

    # Clean up session by releasing
    client.release_session_sync(session_id)
    assert not server.store.has_session(session_id)


def test_gateway_multi_shard_prefix_routing():
    """Verify PrefixRouter routes requests to the correct shard and stamps X-FrontierSplit-Shard-ID."""
    from frontiersplit.context.routing import PrefixRouter

    torch.manual_seed(42)
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()

    server_a = ContextServer()
    port_a = server_a.start_in_thread(host="127.0.0.1", port=0)
    server_b = ContextServer()
    port_b = server_b.start_in_thread(host="127.0.0.1", port=0)

    router = PrefixRouter()
    node_a = router.add_shard("shard-alpha", "127.0.0.1", port_a)
    node_b = router.add_shard("shard-beta", "127.0.0.1", port_b)

    disagg = DisaggregatedModel(model=model, context_client=node_a.get_client())

    app = create_gateway_app(
        model_name="test-routed-llama",
        disaggregated_model=disagg,
        context_client=node_a.get_client(),
        prefix_router=router,
    )
    test_client = TestClient(app)

    try:
        # Telemetry reports both shards
        telem = test_client.get("/v1/telemetry").json()
        assert telem["multi_shard_routing"] is True
        assert set(telem["shards"]) == {"shard-alpha", "shard-beta"}

        # Route request
        resp = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-routed-llama",
                "messages": [{"role": "user", "content": "10 20 30 40"}],
                "max_tokens": 2,
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200
        routed_shard = resp.headers.get("X-FrontierSplit-Shard-ID")
        assert routed_shard in ("shard-alpha", "shard-beta")
        assert resp.headers.get("X-FrontierSplit-Session-ID") is not None
    finally:
        node_a.close()
        node_b.close()
        server_a.stop_thread()
        server_b.stop_thread()


def test_concurrent_gateway_requests_isolated_shards():
    """Verify that concurrent requests routed to different shards do not overwrite each other's client.

    Review finding P1.1: Concurrent requests rewrote context_client on the shared model and patchers,
    causing request A to resume on request B's shard, resulting in 'No prompt KV cache found'
    and uncleaned sessions on the wrong shard.
    """
    import concurrent.futures

    from frontiersplit.context.routing import PrefixRouter

    torch.manual_seed(42)
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()

    server_a = ContextServer()
    port_a = server_a.start_in_thread(host="127.0.0.1", port=0)
    server_b = ContextServer()
    port_b = server_b.start_in_thread(host="127.0.0.1", port=0)

    router = PrefixRouter()
    node_a = router.add_shard("shard-alpha", "127.0.0.1", port_a)
    node_b = router.add_shard("shard-beta", "127.0.0.1", port_b)

    disagg = DisaggregatedModel(model=model, context_client=node_a.get_client())

    app = create_gateway_app(
        model_name="test-concurrent-llama",
        disaggregated_model=disagg,
        context_client=node_a.get_client(),
        prefix_router=router,
    )
    test_client = TestClient(app)

    try:
        # Find two session IDs that naturally hash to shard-alpha and shard-beta
        session_a: str | None = None
        session_b: str | None = None
        for i in range(200):
            sid = f"concurrent-session-{i}"
            shard = router.ring.get_node(f"session:{sid}").shard_id
            if shard == "shard-alpha" and session_a is None:
                session_a = sid
            elif shard == "shard-beta" and session_b is None:
                session_b = sid
            if session_a and session_b:
                break

        assert session_a is not None and session_b is not None

        def make_request(session_id: str, content: str):
            return test_client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-concurrent-llama",
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 4,
                    "temperature": 0.0,
                    "stream": True,
                },
                headers={
                    "X-FrontierSplit-Session-ID": session_id,
                },
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_a = executor.submit(make_request, session_a, "1 2 3 4 5")
            fut_b = executor.submit(make_request, session_b, "10 20 30 40 50")
            resp_a = fut_a.result()
            resp_b = fut_b.result()

        assert resp_a.status_code == 200, f"Request A failed: {resp_a.text}"
        assert resp_b.status_code == 200, f"Request B failed: {resp_b.text}"
        assert resp_a.headers.get("X-FrontierSplit-Shard-ID") == "shard-alpha"
        assert resp_b.headers.get("X-FrontierSplit-Shard-ID") == "shard-beta"

        # Verify stream content contains chunks without error
        assert "data: [DONE]" in resp_a.text
        assert "data: [DONE]" in resp_b.text

        # Verify session_a is retained on server_a only, and session_b on server_b only
        assert server_a.store.has_session(session_a)
        assert not server_a.store.has_session(session_b)
        assert server_b.store.has_session(session_b)
        assert not server_b.store.has_session(session_a)

    finally:
        node_a.close()
        node_b.close()
        server_a.stop_thread()
        server_b.stop_thread()


def test_gateway_with_real_chat_tokenizer_and_eos_filtering():
    """Verify that gateway handles real PreTrainedTokenizerFast (BatchEncoding) and filters EOS tokens."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    # Construct a real fast BPE tokenizer with vocabulary
    raw_tok = Tokenizer(models.BPE())
    raw_tok.pre_tokenizer = pre_tokenizers.Whitespace()
    vocab = {"<eos>": 0, "hello": 1, "world": 2, "user:": 3, "assistant:": 4}
    raw_tok.add_tokens(list(vocab.keys()))

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=raw_tok,
        eos_token="<eos>",
    )
    fast_tok.chat_template = (
        "{% for m in messages %}{{ m.role + ': ' + m.content + '\n' }}{% endfor %}"
        "{% if add_generation_prompt %}assistant:\n{% endif %}"
    )

    # Tiny Llama model with vocab_size 128
    torch.manual_seed(42)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        eos_token_id=0,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()

    server = ContextServer()
    port = server.start_in_thread(host="127.0.0.1", port=0)
    client = ContextClient(host="127.0.0.1", port=port)
    disagg_model = DisaggregatedModel(model=model, context_client=client)

    app = create_gateway_app(
        model_name="test-fast-tok-llama",
        disaggregated_model=disagg_model,
        context_client=client,
        tokenizer=fast_tok,
    )
    test_client = TestClient(app)

    try:
        # 1. Non-streaming request
        resp = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-fast-tok-llama",
                "messages": [{"role": "user", "content": "hello world"}],
                "max_tokens": 8,
                "temperature": 0.0,
                "stream": False,
            },
        )
        assert resp.status_code == 200, f"Failed: {resp.text}"
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        # Special token "<eos>" MUST NOT appear in the user-visible content
        assert "<eos>" not in content

        # 2. Streaming request
        stream_resp = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-fast-tok-llama",
                "messages": [{"role": "user", "content": "hello world"}],
                "max_tokens": 8,
                "temperature": 0.0,
                "stream": True,
            },
        )
        assert stream_resp.status_code == 200
        # Parse streaming SSE chunks
        for line in stream_resp.text.split("\n"):
            if line.startswith("data: ") and not line.endswith("[DONE]"):
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0].get("delta", {})
                delta_content = delta.get("content", "")
                assert "<eos>" not in delta_content

    finally:
        client.close_sync()
        server.stop_thread()


def test_gateway_concurrent_same_session_serialization(disagg_cluster) -> None:
    """Verify that concurrent requests sharing the same session ID are serialized by the gateway."""
    import asyncio
    import httpx

    test_client, _, _ = disagg_cluster
    app = test_client.app

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:

            async def _send(msg: str):
                return await ac.post(
                    "/v1/chat/completions",
                    headers={"X-FrontierSplit-Session-ID": "shared-gateway-sess"},
                    json={
                        "model": "test-disagg-llama",
                        "messages": [{"role": "user", "content": msg}],
                        "max_tokens": 6,
                        "temperature": 0.0,
                        "stream": False,
                    },
                )

            resp1, resp2 = await asyncio.gather(
                _send("1 2 3 4 5"), _send("6 7 8 9 10")
            )
            assert resp1.status_code == 200, f"Request 1 failed: {resp1.text}"
            assert resp2.status_code == 200, f"Request 2 failed: {resp2.text}"

    asyncio.run(_run())


def test_gateway_disconnect_anyio_scope_releases_session(disagg_cluster) -> None:
    """Verify that when a client stream is interrupted/cancelled, AnyIO scope protection releases the session."""
    test_client, _, server = disagg_cluster

    # Request streaming with release_session enabled
    with test_client.stream(
        "POST",
        "/v1/chat/completions",
        headers={
            "X-FrontierSplit-Session-ID": "disconnect-sess",
            "X-FrontierSplit-Release-Session": "true",
        },
        json={
            "model": "test-disagg-llama",
            "messages": [{"role": "user", "content": "1 2 3 4 5"}],
            "max_tokens": 10,
            "temperature": 0.0,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        # Read only the first chunk, then close/exit the context manager early (simulating client disconnect)
        for line in response.iter_lines():
            if line.startswith("data: "):
                break

    # Context manager exit aborts the stream and triggers generator aclose() / AnyIO cancellation.
    # The session must be cleanly evicted on the server:
    assert not server.store.has_session("disconnect-sess")


def test_gateway_early_disconnect_and_registry_cleanup(disagg_cluster) -> None:
    """Verify that early disconnect before reading chunks releases the lock and cleans up registry without leaks."""
    from frontiersplit.gateway import _session_locks

    test_client, _, server = disagg_cluster
    sess_id = "early-dc-sess"

    # Immediately close the stream without reading anything
    with test_client.stream(
        "POST",
        "/v1/chat/completions",
        headers={
            "X-FrontierSplit-Session-ID": sess_id,
        },
        json={
            "model": "test-disagg-llama",
            "messages": [{"role": "user", "content": "1 2 3"}],
            "max_tokens": 5,
            "stream": True,
        },
    ):
        pass

    # The registry must not leak the lock entry
    assert sess_id not in _session_locks

    # Subsequent request using the same session must succeed without being blocked
    resp = test_client.post(
        "/v1/chat/completions",
        headers={"X-FrontierSplit-Session-ID": sess_id},
        json={
            "model": "test-disagg-llama",
            "messages": [{"role": "user", "content": "1 2 3"}],
            "max_tokens": 3,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert sess_id not in _session_locks


def test_gateway_non_streaming_error_no_double_release(disagg_cluster) -> None:
    """Verify that exceptions during non-streaming generation do not cause double-release crashes."""
    from unittest.mock import patch
    from frontiersplit.gateway import _session_locks

    test_client, client, server = disagg_cluster
    sess_id = "error-sess"

    disagg = test_client.app.state.disaggregated_model

    async def _failing_generate(*args, **kwargs):
        raise ValueError("Simulated model forward failure")

    with patch.object(disagg, "generate_async", side_effect=_failing_generate):
        try:
            test_client.post(
                "/v1/chat/completions",
                headers={"X-FrontierSplit-Session-ID": sess_id},
                json={
                    "model": "test-disagg-llama",
                    "messages": [{"role": "user", "content": "1 2 3"}],
                    "max_tokens": 3,
                    "stream": False,
                },
            )
        except ValueError:
            pass

    # Registry must be cleaned up and lock must not be held
    assert sess_id not in _session_locks


