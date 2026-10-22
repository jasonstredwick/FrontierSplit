"""Tests for Disaggregated Delta Prefill, Zero-Copy Prefix Slicing, and Sliding TTL.

Validates:
1. ContextStore and ContextServer sliding TTL tracking and eviction.
2. Zero-copy prefix slicing in attention calculations (mathematical equivalence).
3. Delta KV appending over TCP (sync and async).
4. Multi-turn conversation continuation via Disaggregated Delta Prefill producing
   bit-for-bit identical tokens to monolithic HuggingFace generation.
"""

from __future__ import annotations

import asyncio
import copy
import time

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from frontiersplit.context.context_server import (
    ContextClient,
    ContextServer,
    ContextStore,
)
from frontiersplit.context.hf_model import DisaggregatedModel
from frontiersplit.context.online_softmax import (
    compute_partial_attention,
    finalize_attention,
)


def test_context_store_sliding_ttl_and_eviction() -> None:
    """Verifies that ContextStore touches sessions on access and evicts expired sessions."""
    store = ContextStore(ttl_seconds=0.1)

    k = torch.randn(1, 2, 10, 32)
    v = torch.randn(1, 2, 10, 32)

    store.register_prompt("session-1", 0, k, v)
    assert store.has_session("session-1")
    t1 = store.get_last_access("session-1")
    assert t1 is not None

    # Accessing prompt updates timestamp
    time.sleep(0.02)
    store.get_prompt("session-1", 0)
    t2 = store.get_last_access("session-1")
    assert t2 is not None and t2 > t1

    # Appending delta KV updates timestamp
    time.sleep(0.02)
    k_delta = torch.randn(1, 2, 5, 32)
    v_delta = torch.randn(1, 2, 5, 32)
    new_len = store.append_kv("session-1", 0, k_delta, v_delta)
    assert new_len == 15
    t3 = store.get_last_access("session-1")
    assert t3 is not None and t3 > t2

    # Verify no eviction before TTL
    evicted = store.evict_expired(max_idle_seconds=10.0)
    assert len(evicted) == 0
    assert store.has_session("session-1")

    # Wait for TTL to expire
    time.sleep(0.12)
    evicted = store.evict_expired()
    assert "session-1" in evicted
    assert not store.has_session("session-1")


def test_context_server_zero_copy_prefix_slicing() -> None:
    """Verifies that query_partial_attention with context_len matches full attention on sliced tensors."""
    server = ContextServer()
    torch.manual_seed(42)

    batch_size = 1
    num_heads = 4
    num_kv_heads = 2
    full_len = 50
    slice_len = 20
    head_dim = 32

    k_full = torch.randn(batch_size, num_kv_heads, full_len, head_dim)
    v_full = torch.randn(batch_size, num_kv_heads, full_len, head_dim)
    q = torch.randn(batch_size, num_heads, 1, head_dim)

    session_id = "test-slice-session"
    server.register_prompt(session_id, 0, k_full, v_full)

    # 1. Query with context_len=slice_len
    chunk_sliced = server.query_partial_attention(
        session_id=session_id,
        layer_idx=0,
        q=q,
        context_len=slice_len,
    )
    out_sliced = finalize_attention(chunk_sliced)

    # 2. Compute expected attention directly against k_full[:, :, :slice_len]
    expected_chunk = compute_partial_attention(
        q=q,
        k=k_full[:, :, :slice_len, :],
        v=v_full[:, :, :slice_len, :],
    )
    out_expected = finalize_attention(expected_chunk)

    torch.testing.assert_close(
        out_sliced,
        out_expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_context_server_delta_append_kv_tcp() -> None:
    """Verifies appending delta KV over TCP via ContextClient (sync and async)."""
    server = ContextServer()
    port = server.start_in_thread(host="127.0.0.1", port=0)
    client = ContextClient(host="127.0.0.1", port=port)

    try:
        session_id = "test-append-session"
        k_base = torch.randn(1, 2, 10, 32)
        v_base = torch.randn(1, 2, 10, 32)

        # Register base prompt
        assert client.register_prompt_sync(session_id, 0, k_base, v_base)

        # Append delta synchronously
        k_delta1 = torch.randn(1, 2, 6, 32)
        v_delta1 = torch.randn(1, 2, 6, 32)
        new_len1 = client.append_kv_sync(session_id, 0, k_delta1, v_delta1)
        assert new_len1 == 16

        # Query partial attention over extended cache
        q = torch.randn(1, 4, 1, 32)
        chunk = client.query_partial_attention_sync(session_id, 0, q)
        assert chunk.accumulator.shape == (1, 4, 1, 32)

        # Verify that context_len slice still works over TCP
        chunk_sliced = client.query_partial_attention_sync(
            session_id, 0, q, context_len=10
        )
        assert chunk_sliced.accumulator.shape == (1, 4, 1, 32)

    finally:
        client.close_sync()
        server.stop_thread()


@pytest.fixture
def tiny_llama() -> tuple[LlamaForCausalLM, torch.Tensor, torch.Tensor]:
    """Provides a tiny Llama model and 2-turn prompts."""
    torch.manual_seed(1234)
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

    prompt_turn1 = torch.tensor([[5, 12, 45, 99, 21]])
    # Turn 2 extends Turn 1 prompt with 2 additional tokens
    prompt_turn2 = torch.tensor([[5, 12, 45, 99, 21, 64, 88]])
    return model, prompt_turn1, prompt_turn2


def test_disaggregated_model_delta_prefill_e2e(
    tiny_llama: tuple[LlamaForCausalLM, torch.Tensor, torch.Tensor],
) -> None:
    """Verifies that multi-turn Disaggregated Delta Prefill produces bit-for-bit identical tokens to monolithic model."""
    model, prompt_turn1, prompt_turn2 = tiny_llama
    num_tokens = 5

    # Compute expected monolithic generation for Turn 2
    with torch.no_grad():
        out_mono = model(prompt_turn2, use_cache=True)
        mono_cache = copy.deepcopy(out_mono.past_key_values)
        next_tok = out_mono.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        expected_tokens = [next_tok.item()]
        cur_tok = next_tok
        for _ in range(num_tokens - 1):
            out = model(cur_tok, past_key_values=mono_cache, use_cache=True)
            cur_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            expected_tokens.append(cur_tok.item())

    # Set up Disaggregated Context Server
    server = ContextServer()
    port = server.start_in_thread(host="127.0.0.1", port=0)
    client = ContextClient(host="127.0.0.1", port=port)

    try:
        disagg_model = DisaggregatedModel(model=model, context_client=client)

        # Turn 1: Cold prefill and generation with release_on_completion=False (retains KV with sliding TTL)
        session_id = "agent-conversation-1"
        disagg_model.generate(
            input_ids=prompt_turn1,
            max_new_tokens=3,
            session_id=session_id,
            release_on_completion=False,
        )

        # Confirm Turn 1 prompt is cached
        assert server.store.has_session(session_id)

        # Turn 2: Follow-up message sharing first 5 tokens + 2 new tokens
        # Must execute Disaggregated Delta Prefill!
        turn2_tokens = disagg_model.generate(
            input_ids=prompt_turn2,
            max_new_tokens=num_tokens,
            session_id=session_id,
            use_prefix_cache=True,
            release_on_completion=True,
        )

        assert turn2_tokens == expected_tokens, (
            f"Multi-turn delta prefill mismatch!\n"
            f"Expected:     {expected_tokens}\n"
            f"Disaggregated: {turn2_tokens}"
        )

        # Verify RadixPrefixCache has exact token count (no multi-layer token_ids duplication)
        assert server.prefix_cache.total_tokens == prompt_turn2.shape[1]

        # With release_on_completion=True, session is cleaned up
        assert not server.store.has_session(session_id)

    finally:
        client.close_sync()
        server.stop_thread()


def test_disaggregated_model_fp16_delta_prefill(
    tiny_llama: tuple[LlamaForCausalLM, torch.Tensor, torch.Tensor],
) -> None:
    """Verifies that FP16 models keep KV caches in FP16 while executing Delta Prefill and Online Softmax in FP32."""
    base_model, prompt_turn1, prompt_turn2 = tiny_llama
    fp16_model = copy.deepcopy(base_model).to(torch.float16)
    fp16_model.eval()

    server = ContextServer()
    port = server.start_in_thread(host="127.0.0.1", port=0)
    client = ContextClient(host="127.0.0.1", port=port)

    try:
        disagg_model = DisaggregatedModel(model=fp16_model, context_client=client)
        session_id = "fp16-delta-session"

        # Turn 1: Cold prefill in FP16
        disagg_model.generate(
            input_ids=prompt_turn1,
            max_new_tokens=2,
            session_id=session_id,
            release_on_completion=False,
        )

        # Verify KV cache on Context Server is stored in FP16
        kv_layer0 = server.store.get_prompt(session_id, 0)
        assert kv_layer0 is not None
        assert kv_layer0[0].dtype == torch.float16
        assert kv_layer0[1].dtype == torch.float16

        # Turn 2: Multi-token Delta Prefill in FP16 (tests causal mask & FP32 softmax upcast)
        turn2_tokens = disagg_model.generate(
            input_ids=prompt_turn2,
            max_new_tokens=3,
            session_id=session_id,
            use_prefix_cache=True,
            release_on_completion=False,
        )
        assert len(turn2_tokens) == 3

        # Verify appended KV cache remains FP16 and exact length
        kv_after = server.store.get_prompt(session_id, 0)
        assert kv_after is not None
        assert kv_after[0].dtype == torch.float16
        assert kv_after[0].shape[-2] == prompt_turn2.shape[1]
        assert server.prefix_cache.total_tokens == prompt_turn2.shape[1]

    finally:
        client.close_sync()
        server.stop_thread()


def test_disaggregated_model_async_delta_prefill_e2e(
    tiny_llama: tuple[LlamaForCausalLM, torch.Tensor, torch.Tensor],
) -> None:
    """Verifies that asynchronous multi-turn Delta Prefill matches monolithic baseline."""
    model, prompt_turn1, prompt_turn2 = tiny_llama
    num_tokens = 5

    # Compute expected monolithic generation for Turn 2
    with torch.no_grad():
        out_mono = model(prompt_turn2, use_cache=True)
        mono_cache = copy.deepcopy(out_mono.past_key_values)
        next_tok = out_mono.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        expected_tokens = [next_tok.item()]
        cur_tok = next_tok
        for _ in range(num_tokens - 1):
            out = model(cur_tok, past_key_values=mono_cache, use_cache=True)
            cur_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            expected_tokens.append(cur_tok.item())

    async def _run() -> None:
        server = ContextServer()
        port = server.start_in_thread(host="127.0.0.1", port=0)
        client = ContextClient(host="127.0.0.1", port=port)

        try:
            disagg_model = DisaggregatedModel(model=model, context_client=client)

            session_id = "agent-async-turn"
            await disagg_model.generate_async(
                input_ids=prompt_turn1,
                max_new_tokens=3,
                session_id=session_id,
                release_on_completion=False,
            )

            assert server.store.has_session(session_id)

            turn2_tokens = await disagg_model.generate_async(
                input_ids=prompt_turn2,
                max_new_tokens=num_tokens,
                session_id=session_id,
                use_prefix_cache=True,
                release_on_completion=True,
            )

            assert turn2_tokens == expected_tokens
            assert not server.store.has_session(session_id)

        finally:
            await client.close()
            server.stop_thread()

    asyncio.run(_run())
