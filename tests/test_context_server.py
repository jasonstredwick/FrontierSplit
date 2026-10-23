"""Unit tests for the Disaggregated Context Server and Client.

Validates that static prompt KV caches can be registered and queried over raw
TCP sockets to yield exact partial attention calculations.
"""

from __future__ import annotations

import asyncio
import socket

import pytest
import torch

from frontiersplit.context.context_server import (
    ContextClient,
    ContextServer,
    ContextStore,
    deserialize_chunk,
    deserialize_tensor,
    serialize_chunk,
    serialize_tensor,
)
from frontiersplit.context.online_softmax import (
    PartialAttentionChunk,
    compute_partial_attention,
    finalize_attention,
    merge_two_partial_attentions,
)
from frontiersplit.context.prefix_cache import RadixPrefixCache


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


def test_raw_binary_serialization_roundtrip():
    """Verify zero-pickle raw binary tensor and chunk serialization across dtypes."""
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int64):
        if dtype == torch.int64:
            t = torch.randint(-100, 100, (2, 4, 8), dtype=dtype)
        else:
            t = torch.randn(1, 8, 16, 64, dtype=dtype)
        raw = serialize_tensor(t)
        # Ensure it's compact raw binary (not a PK/ZIP archive)
        assert not raw.startswith(b"PK")
        rec = deserialize_tensor(raw)
        assert rec.dtype == t.dtype
        assert torch.equal(rec, t)

    chunk = PartialAttentionChunk(
        accumulator=torch.randn(1, 8, 1, 64, dtype=torch.float32),
        max_score=torch.randn(1, 8, 1, 1, dtype=torch.float32),
        sum_exp=torch.rand(1, 8, 1, 1, dtype=torch.float32),
        dtype=torch.float16,
    )
    chunk_bytes = serialize_chunk(chunk)
    assert not chunk_bytes.startswith(b"PK")
    rec_chunk = deserialize_chunk(chunk_bytes)
    assert rec_chunk.dtype == torch.float16
    assert torch.equal(rec_chunk.accumulator, chunk.accumulator)
    assert torch.equal(rec_chunk.max_score, chunk.max_score)
    assert torch.equal(rec_chunk.sum_exp, chunk.sum_exp)


def test_context_server_session_pinning_and_unpinning():
    """Verify active sessions pin RadixNodes against LRU eviction and unpin on release."""
    server = ContextServer(max_cached_tokens=10)
    t1 = [1, 2, 3, 4, 5, 6]
    k1 = torch.randn(1, 2, 6, 16)
    v1 = torch.randn(1, 2, 6, 16)

    # Register s1 (6 tokens) -> pinned while s1 is active
    server.register_prompt("s1", layer_idx=0, k=k1, v=v1, token_ids=t1)
    nodes_s1, _ = server.prefix_cache.match(t1)
    assert nodes_s1[0].session_count == 1

    # Release s1 -> unpins node in RadixPrefixCache while keeping it cached
    server.release_session("s1")
    assert nodes_s1[0].session_count == 0
    assert server.prefix_cache.total_tokens == 6


def test_incomplete_multilayer_upload_not_exposed_and_backfilled():
    """Verify multi-layer uploads do not expose incomplete caches and backfill bound sessions."""
    server = ContextServer(expected_num_layers=2)
    tokens = [10, 20, 30, 40, 50]
    k0 = torch.randn(1, 2, 5, 16)
    v0 = torch.randn(1, 2, 5, 16)
    k1 = torch.randn(1, 2, 5, 16)
    v1 = torch.randn(1, 2, 5, 16)
    q = torch.randn(1, 2, 1, 16)

    # Upload only layer 0 with token_ids -> should NOT be matchable yet because layer 1 is missing
    server.register_prompt("upload-sess", layer_idx=0, k=k0, v=v0, token_ids=tokens)
    m_len, bound = server.match_prefix(tokens, target_session_id="reader-sess")
    assert m_len == 0
    assert bound is None

    # Upload layer 1 -> now complete and matchable for all layers
    server.register_prompt("upload-sess", layer_idx=1, k=k1, v=v1, token_ids=tokens)
    m_len, bound = server.match_prefix(tokens, target_session_id="reader-sess")
    assert m_len == 5
    assert bound == "reader-sess"

    # Querying both layer 0 and layer 1 on the bound session succeeds
    c0 = server.query_partial_attention("reader-sess", layer_idx=0, q=q)
    c1 = server.query_partial_attention("reader-sess", layer_idx=1, q=q)
    assert c0.accumulator.shape == (1, 2, 1, 16)
    assert c1.accumulator.shape == (1, 2, 1, 16)


def test_async_rpc_cancellation_does_not_corrupt_subsequent_calls():
    """Verify cancelling an in-flight async RPC resets the socket so the next call gets its own response."""

    async def _run() -> None:
        port = get_free_port()
        server = ContextServer()
        await server.start_server(host="127.0.0.1", port=port)
        client = ContextClient(host="127.0.0.1", port=port)

        try:
            tokens_full = [1, 2, 3, 4, 5, 6, 7, 8]
            k = torch.randn(1, 2, 8, 16)
            v = torch.randn(1, 2, 8, 16)
            await client.register_prompt(
                "seed-sess", layer_idx=0, k=k, v=v, token_ids=tokens_full
            )

            # Intercept read_context_frame_async once to simulate cancellation while reply is in flight
            orig_read = client.connect
            await orig_read()
            assert client.writer is not None and client.reader is not None

            # Start a match_prefix request for 3 tokens ("cancelled") and cancel before reading response
            from frontiersplit.context.context_server import (
                MSG_MATCH_PREFIX_REQ,
                pack_context_frame,
            )

            stale_frame = pack_context_frame(
                MSG_MATCH_PREFIX_REQ,
                {"token_ids": [1, 2, 3], "session_id": "cancelled"},
                b"",
            )
            client.writer.write(stale_frame)
            await client.writer.drain()
            # Simulate task cancellation handler resetting transport
            client._reset_async_transport()

            # Subsequent match_prefix for 8 tokens ("live") must return (8, "live"), NOT (3, "cancelled")
            matched_len, sid = await client.match_prefix(tokens_full, session_id="live")
            assert (matched_len, sid) == (8, "live")
        finally:
            await client.close()
            await server.stop_server()

    asyncio.run(_run())


def test_context_client_sync_and_async_timeout():
    """Verify ContextClient.timeout is enforced on stalled sync and async connections."""
    import time

    stalled_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    stalled_server.bind(("127.0.0.1", 0))
    stalled_server.listen(4)
    port = stalled_server.getsockname()[1]

    accepted_socks: list[socket.socket] = []

    async def _run() -> None:
        client = ContextClient(host="127.0.0.1", port=port, timeout=0.03)
        try:
            t0 = time.monotonic()
            with pytest.raises((TimeoutError, OSError)):
                client.match_prefix_sync([1, 2, 3])
            elapsed_sync = time.monotonic() - t0
            assert elapsed_sync < 0.5
            assert client._sync_sock is None

            t1 = time.monotonic()
            with pytest.raises(TimeoutError):
                await client.match_prefix([1, 2, 3])
            elapsed_async = time.monotonic() - t1
            assert elapsed_async < 0.5
            assert client.writer is None
        finally:
            await client.close()

    try:
        asyncio.run(_run())
    finally:
        for s in accepted_socks:
            s.close()
        stalled_server.close()


def test_context_store_zero_copy_chunks_registration():
    """Verify that ContextStore stores and appends chunks without memory copies."""
    store = ContextStore()
    k1 = torch.randn(1, 4, 16, 32)
    v1 = torch.randn(1, 4, 16, 32)
    k2 = torch.randn(1, 4, 32, 32)
    v2 = torch.randn(1, 4, 32, 32)

    store.register_chunks("sess-zc", 0, [(k1, v1), (k2, v2)])
    chunks = store.get_chunks("sess-zc", 0)
    assert chunks is not None
    assert len(chunks) == 2

    # Zero-copy verification: memory address pointers must match exactly
    assert chunks[0][0].data_ptr() == k1.data_ptr()
    assert chunks[0][1].data_ptr() == v1.data_ptr()
    assert chunks[1][0].data_ptr() == k2.data_ptr()
    assert chunks[1][1].data_ptr() == v2.data_ptr()

    # O(1) list append delta: must not reallocate existing chunks
    k3 = torch.randn(1, 4, 8, 32)
    v3 = torch.randn(1, 4, 8, 32)
    new_len = store.append_kv("sess-zc", 0, k3, v3)
    assert new_len == 16 + 32 + 8

    updated_chunks = store.get_chunks("sess-zc", 0)
    assert updated_chunks is not None
    assert len(updated_chunks) == 3
    assert updated_chunks[0][0].data_ptr() == k1.data_ptr()
    assert updated_chunks[1][0].data_ptr() == k2.data_ptr()
    assert updated_chunks[2][0].data_ptr() == k3.data_ptr()

    # Concatenated prompt fallback
    prompt = store.get_prompt("sess-zc", 0)
    assert prompt is not None
    assert prompt[0].shape[-2] == 56


def test_context_server_zero_copy_radix_match_and_chunk_attention():
    """Verify that match_prefix shares RadixNode memory zero-copy and chunk attention is bit-exact."""
    server = ContextServer()
    tokens1 = [10, 20, 30]
    tokens2 = [10, 20, 30, 40, 50]

    k_prompt = torch.randn(1, 4, 5, 32)
    v_prompt = torch.randn(1, 4, 5, 32)

    # Insert prefix tree with two branching nodes
    server.register_prompt(
        "sess-base",
        0,
        k_prompt[:, :, :3, :],
        v_prompt[:, :, :3, :],
        token_ids=tokens1,
        num_layers=1,
    )
    server.register_prompt(
        "sess-ext", 0, k_prompt, v_prompt, token_ids=tokens2, num_layers=1
    )

    # Match full prefix of 5 tokens: should span 2 nodes in the Radix tree
    matched_len, session_id = server.match_prefix(tokens2)
    assert matched_len == 5
    assert session_id is not None

    chunks = server.store.get_chunks(session_id, 0)
    assert chunks is not None
    assert len(chunks) == 2
    assert chunks[0][0].shape[-2] == 3
    assert chunks[1][0].shape[-2] == 2

    # Query chunk attention with single decode query (q_len=1)
    q = torch.randn(1, 4, 1, 32)
    chunk_res = server.query_partial_attention(session_id, 0, q=q)
    expected_res = compute_partial_attention(q, k_prompt, v_prompt)

    assert torch.allclose(chunk_res.accumulator, expected_res.accumulator, atol=1e-5)
    assert torch.allclose(chunk_res.max_score, expected_res.max_score, atol=1e-5)
    assert torch.allclose(chunk_res.sum_exp, expected_res.sum_exp, atol=1e-5)

    # Query with context_len slicing across chunk boundary (e.g., context_len=4)
    sliced_res = server.query_partial_attention(session_id, 0, q=q, context_len=4)
    expected_sliced = compute_partial_attention(
        q, k_prompt[:, :, :4, :], v_prompt[:, :, :4, :]
    )
    assert torch.allclose(
        sliced_res.accumulator, expected_sliced.accumulator, atol=1e-5
    )
    assert torch.allclose(sliced_res.max_score, expected_sliced.max_score, atol=1e-5)
    assert torch.allclose(sliced_res.sum_exp, expected_sliced.sum_exp, atol=1e-5)

    # Query with sliding window across chunks (e.g. sliding_window=3 -> tokens [2, 3, 4])
    sw_res = server.query_partial_attention(session_id, 0, q=q, sliding_window=3)
    expected_sw = compute_partial_attention(
        q, k_prompt[:, :, 2:, :], v_prompt[:, :, 2:, :]
    )
    assert torch.allclose(sw_res.accumulator, expected_sw.accumulator, atol=1e-5)
    assert torch.allclose(sw_res.max_score, expected_sw.max_score, atol=1e-5)
    assert torch.allclose(sw_res.sum_exp, expected_sw.sum_exp, atol=1e-5)


def test_context_server_delta_eviction_does_not_publish_orphan_prefix():
    """Verify that after eviction, appending to an evicted session never publishes suffix under an unrelated prefix."""
    server = ContextServer(max_cached_tokens=5)

    k1 = torch.randn(1, 2, 3, 16)
    v1 = torch.randn(1, 2, 3, 16)
    # Session 1: [6, 7, 8] (3 tokens)
    server.register_prompt("sess-1", 0, k1, v1, token_ids=[6, 7, 8], num_layers=1)
    # Unpin sess-1 so it can be evicted
    server._clear_session_pin("sess-1")

    # Session 2: [1, 2, 3, 4, 5] (5 tokens). Evicts sess-1 from prefix cache
    k2 = torch.randn(1, 2, 5, 16)
    v2 = torch.randn(1, 2, 5, 16)
    server.register_prompt("sess-2", 0, k2, v2, token_ids=[1, 2, 3, 4, 5], num_layers=1)

    # Verify [6, 7, 8] is evicted from prefix cache
    m_len, _ = server.match_prefix([6, 7, 8])
    assert m_len == 0

    # Session 1 appends [9]
    k_delta = torch.randn(1, 2, 1, 16)
    v_delta = torch.randn(1, 2, 1, 16)
    server.append_kv("sess-1", 0, k_delta, v_delta, token_ids=[9], num_layers=1)

    # An unrelated prompt [9] MUST NOT match!
    orphan_len, _ = server.match_prefix([9])
    assert orphan_len == 0


def test_append_without_tokens_and_omitted_upload_id():
    """Verify that append_kv without token metadata correctly commits to ContextStore and does not discard successive appends."""
    server = ContextServer()
    k5 = torch.randn(1, 2, 5, 16)
    v5 = torch.randn(1, 2, 5, 16)
    server.register_prompt("sess-defaults", 0, k5, v5)
    assert server.store.get_seq_len("sess-defaults", 0) == 5

    k2 = torch.randn(1, 2, 2, 16)
    v2 = torch.randn(1, 2, 2, 16)
    ret1 = server.append_kv("sess-defaults", 0, k2, v2)
    assert ret1 == 7
    assert server.store.get_seq_len("sess-defaults", 0) == 7

    k2_second = torch.randn(1, 2, 2, 16)
    v2_second = torch.randn(1, 2, 2, 16)
    ret2 = server.append_kv("sess-defaults", 0, k2_second, v2_second)
    assert ret2 == 9
    assert server.store.get_seq_len("sess-defaults", 0) == 9


def test_canonical_radix_storage_sharing():
    """Verify zero-copy canonical storage sharing between session store and prefix cache."""
    server = ContextServer()
    tokens = [10, 20, 30, 40]
    k = torch.randn(1, 2, 4, 16)
    v = torch.randn(1, 2, 4, 16)
    server.register_prompt("sess-shared", 0, k, v, token_ids=tokens, num_layers=1)

    nodes, matched = server.prefix_cache.match(tokens)
    assert matched == 4
    assert len(nodes) == 1
    radix_k = nodes[0].k_cache[0]

    chunks = server.store.get_chunks("sess-shared", 0)
    assert chunks is not None and len(chunks) == 1
    store_k, _ = chunks[0]

    # Verify identical underlying storage (zero memory duplicate)
    assert store_k.data_ptr() == radix_k.data_ptr()


def test_gap_in_token_metadata_preserves_live_cache_and_prohibits_invalid_suffix():
    """Verify that unindexed appends between indexed ones preserve the live KV store without truncation or publishing invalid prefix."""
    server = ContextServer()
    # 1. Register 5 indexed tokens
    k5 = torch.randn(1, 2, 5, 16)
    v5 = torch.randn(1, 2, 5, 16)
    server.register_prompt(
        "sess-gap", 0, k5, v5, token_ids=[1, 2, 3, 4, 5], num_layers=1
    )
    assert server.store.get_seq_len("sess-gap", 0) == 5

    # 2. Append 2 tokens without metadata
    k2 = torch.randn(1, 2, 2, 16)
    v2 = torch.randn(1, 2, 2, 16)
    server.append_kv("sess-gap", 0, k2, v2)
    assert server.store.get_seq_len("sess-gap", 0) == 7

    # 3. Append 1 token with metadata
    k1 = torch.randn(1, 2, 1, 16)
    v1 = torch.randn(1, 2, 1, 16)
    server.append_kv("sess-gap", 0, k1, v1, token_ids=[8], num_layers=1)

    # Cache length must reach 8 (5 -> 7 -> 8), NOT truncated to 6!
    assert server.store.get_seq_len("sess-gap", 0) == 8

    # Suffix [8] must NOT be published as a prefix attached directly to [1, 2, 3, 4, 5]
    matched, _ = server.match_prefix([1, 2, 3, 4, 5, 8])
    assert matched == 5  # Only matches [1..5], does not match 8 because of the gap


def test_model_identity_isolation_on_shared_server():
    """Verify that multiple models with distinct model_ids sharing a server have isolated prefix caches."""
    server = ContextServer()
    tokens = [10, 20, 30]
    k_a = torch.randn(1, 2, 3, 16)
    v_a = torch.randn(1, 2, 3, 16)
    server.register_prompt(
        "sess-model-a",
        0,
        k_a,
        v_a,
        token_ids=tokens,
        num_layers=1,
        model_id="model-a",
    )

    # Model B queries prefix for exact same tokens
    matched_len, _ = server.match_prefix(tokens, model_id="model-b")
    assert matched_len == 0

    # Model A queries prefix
    matched_len_a, _ = server.match_prefix(tokens, model_id="model-a")
    assert matched_len_a == 3


def test_server_side_session_lease_enforcement():
    """Verify exclusive session lease ownership and prevention of overlapping access."""
    server = ContextServer()
    # Acquire lease for lease-1
    assert server.acquire_lease("sess-lease", "lease-1") is True
    # Duplicate lease with same lease_id succeeds
    assert server.acquire_lease("sess-lease", "lease-1") is True
    # Different lease_id fails
    assert server.acquire_lease("sess-lease", "lease-2") is False

    # Calling register_prompt with mismatched lease raises RuntimeError
    k = torch.randn(1, 2, 4, 16)
    v = torch.randn(1, 2, 4, 16)
    with pytest.raises(
        RuntimeError,
        match="Session 'sess-lease' is leased by active generation 'lease-1'",
    ):
        server.register_prompt("sess-lease", 0, k, v, lease_id="lease-2")

    # Releasing with wrong lease fails
    assert server.release_lease("sess-lease", "lease-2") is False
    # Releasing with correct lease succeeds
    assert server.release_lease("sess-lease", "lease-1") is True
    # Now lease-2 can be acquired
    assert server.acquire_lease("sess-lease", "lease-2") is True


def test_multilayer_registration_with_optional_defaults():
    """Verify that registering multiple layers with optional defaults does not delete earlier layers."""
    server = ContextServer()
    k0 = torch.randn(1, 2, 4, 16)
    v0 = torch.randn(1, 2, 4, 16)
    k1 = torch.randn(1, 2, 4, 16)
    v1 = torch.randn(1, 2, 4, 16)

    # Register layer 0 without num_layers
    server.register_prompt("sess-defaults-multi", 0, k0, v0)
    assert server.store.get_prompt("sess-defaults-multi", 0) is not None

    # Register layer 1 without num_layers
    server.register_prompt("sess-defaults-multi", 1, k1, v1)
    # Both layer 0 and layer 1 must exist
    assert server.store.get_prompt("sess-defaults-multi", 0) is not None
    assert server.store.get_prompt("sess-defaults-multi", 1) is not None

