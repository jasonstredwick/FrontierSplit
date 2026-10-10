"""Unit tests for the Disaggregated Context Server and Client.

Validates that static prompt KV caches can be registered and queried over raw
TCP sockets to yield exact partial attention calculations.
"""

from __future__ import annotations

import asyncio
import socket

import pytest
import torch

from frontiersplit.context_server import (
    ContextClient,
    ContextServer,
    ContextStore,
)
from frontiersplit.online_softmax import (
    compute_partial_attention,
    finalize_attention,
    merge_two_partial_attentions,
)


def get_free_port() -> int:
    """Finds an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_context_store_basic_lifecycle():
    """Verify registration, retrieval, and eviction in ContextStore."""
    store = ContextStore()
    k = torch.randn(1, 8, 32, 64)
    v = torch.randn(1, 8, 32, 64)

    assert not store.has_session("session-1")
    store.register_prompt("session-1", layer_idx=0, k=k, v=v)

    assert store.has_session("session-1")
    assert store.num_sessions == 1

    retrieved = store.get_prompt("session-1", layer_idx=0)
    assert retrieved is not None
    k_ret, v_ret = retrieved
    assert torch.equal(k, k_ret)
    assert torch.equal(v, v_ret)

    # Missing layer returns None
    assert store.get_prompt("session-1", layer_idx=1) is None

    # Release session
    evicted = store.release_session("session-1")
    assert evicted is True
    assert not store.has_session("session-1")
    assert store.num_sessions == 0


def test_context_server_direct_query():
    """Verify direct partial attention calculation through ContextServer."""
    server = ContextServer()
    batch, num_heads, q_len, kv_len, head_dim = 1, 16, 1, 64, 64

    q = torch.randn(batch, num_heads, q_len, head_dim)
    k_prompt = torch.randn(batch, num_heads, kv_len, head_dim)
    v_prompt = torch.randn(batch, num_heads, kv_len, head_dim)

    server.register_prompt("sess-42", layer_idx=2, k=k_prompt, v=v_prompt)

    chunk = server.query_partial_attention("sess-42", layer_idx=2, q=q)
    expected_chunk = compute_partial_attention(q, k_prompt, v_prompt)

    assert torch.allclose(chunk.accumulator, expected_chunk.accumulator, atol=1e-5)
    assert torch.allclose(chunk.max_score, expected_chunk.max_score, atol=1e-5)
    assert torch.allclose(chunk.sum_exp, expected_chunk.sum_exp, atol=1e-5)

    with pytest.raises(KeyError):
        server.query_partial_attention("missing-sess", layer_idx=0, q=q)


def test_context_server_client_tcp_roundtrip():
    """Verify registering and querying partial attention over a live TCP connection."""

    async def _run() -> None:
        port = get_free_port()
        server = ContextServer()
        await server.start_server(host="127.0.0.1", port=port)

        client = ContextClient(host="127.0.0.1", port=port)

        try:
            batch, num_heads, q_len, head_dim = 1, 8, 1, 64
            prompt_len, output_len = 128, 32
            total_len = prompt_len + output_len

            # Create full context tensors
            q = torch.randn(batch, num_heads, q_len, head_dim)
            k_full = torch.randn(batch, num_heads, total_len, head_dim)
            v_full = torch.randn(batch, num_heads, total_len, head_dim)

            k_p = k_full[:, :, :prompt_len, :]
            v_p = v_full[:, :, :prompt_len, :]
            k_o = k_full[:, :, prompt_len:, :]
            v_o = v_full[:, :, prompt_len:, :]

            # 1. Register prompt on remote Context Server via TCP client
            ok = await client.register_prompt("live-session", layer_idx=0, k=k_p, v=v_p)
            assert ok is True

            # 2. Query partial attention from remote server
            remote_chunk = await client.query_partial_attention(
                "live-session", layer_idx=0, q=q
            )

            # 3. Compute local decode output chunk locally
            local_chunk = compute_partial_attention(q, k_o, v_o)

            # 4. Merge via Online Softmax
            merged = merge_two_partial_attentions(remote_chunk, local_chunk)
            final_out = finalize_attention(merged)

            # 5. Compare against monolithic reference
            ref_chunk = compute_partial_attention(q, k_full, v_full)
            ref_out = finalize_attention(ref_chunk)

            assert torch.allclose(ref_out, final_out, atol=1e-5, rtol=1e-5)

            # 6. Evict session over TCP
            evicted = await client.release_session("live-session")
            assert evicted is True
            assert not server.store.has_session("live-session")

        finally:
            await client.close()
            await server.stop_server()

    asyncio.run(_run())


def test_context_server_prefix_caching_over_tcp():
    """Verify that ContextServer indexes token prefixes and allows TCP prefix matching."""
    server = ContextServer()
    port = server.start_in_thread(host="127.0.0.1", port=0)
    client = ContextClient(host="127.0.0.1", port=port)

    try:
        tokens_prompt = [101, 102, 103, 104, 105, 106]
        k_prompt = torch.randn(1, 4, len(tokens_prompt), 32)
        v_prompt = torch.randn(1, 4, len(tokens_prompt), 32)

        # 1. Register prompt with token_ids
        ok = client.register_prompt_sync(
            session_id="first-user-session",
            layer_idx=0,
            k=k_prompt,
            v=v_prompt,
            token_ids=tokens_prompt,
        )
        assert ok is True

        # 2. Query prefix match for a new prompt that shares the first 6 tokens + 2 new tokens
        new_prompt = [101, 102, 103, 104, 105, 106, 999, 1000]
        matched_len, bound_session = client.match_prefix_sync(new_prompt)
        assert matched_len == 6
        assert bound_session is not None

        # 3. Query partial attention using the bound session
        q = torch.randn(1, 4, 1, 32)
        chunk = client.query_partial_attention_sync(
            session_id=bound_session,
            layer_idx=0,
            q=q,
        )
        expected_chunk = compute_partial_attention(q, k_prompt, v_prompt)
        assert torch.allclose(chunk.accumulator, expected_chunk.accumulator, atol=1e-5)

        # 4. Query prefix match for a completely disjoint prompt
        disjoint_prompt = [555, 666, 777]
        m_len, dis_sess = client.match_prefix_sync(disjoint_prompt)
        assert m_len == 0
        assert dis_sess is None

    finally:
        client.close_sync()
        server.stop_thread()
