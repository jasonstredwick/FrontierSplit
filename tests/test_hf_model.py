"""Integration tests for DisaggregatedModel with real HuggingFace architecture.

Validates that wrapping a HuggingFace causal language model with DisaggregatedModel
produces bit-for-bit identical token sequences to standard unchunked generation
both synchronously and asynchronously over TCP.
"""

from __future__ import annotations

import asyncio
import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from frontiersplit.context.context_server import ContextClient, ContextServer
from frontiersplit.context.hf_model import DisaggregatedModel


@pytest.fixture
def llama_model_and_prompt() -> tuple[LlamaForCausalLM, torch.Tensor, list[int]]:
    """Fixture providing a tiny Llama model, prompt tensor, and monolithic baseline tokens."""
    torch.manual_seed(9999)
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

    prompt = torch.tensor([[5, 12, 45, 99, 21]])
    num_tokens_to_generate = 6

    with torch.no_grad():
        out_prefill = model(prompt, use_cache=True)
        mono_cache = copy.deepcopy(out_prefill.past_key_values)
        first_tok = out_prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        mono_generated = [first_tok.item()]
        cur_tok = first_tok
        for _ in range(num_tokens_to_generate - 1):
            out = model(cur_tok, past_key_values=mono_cache, use_cache=True)
            cur_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            mono_generated.append(cur_tok.item())

    return model, prompt, mono_generated


def test_huggingface_disaggregated_model_sync_generation(
    llama_model_and_prompt: tuple[LlamaForCausalLM, torch.Tensor, list[int]],
) -> None:
    """Verify synchronous DisaggregatedModel generates identical tokens over TCP."""
    model, prompt, expected_tokens = llama_model_and_prompt

    server = ContextServer()
    port = server.start_in_thread(host="127.0.0.1", port=0)
    client = ContextClient(host="127.0.0.1", port=port)

    try:
        disagg_model = DisaggregatedModel(model=model, context_client=client)

        disagg_generated = disagg_model.generate(
            input_ids=prompt,
            max_new_tokens=len(expected_tokens),
            session_id="hf-sync-session",
        )

        assert expected_tokens == disagg_generated, (
            f"Generated token mismatch!\n"
            f"Expected:      {expected_tokens}\n"
            f"Disaggregated: {disagg_generated}"
        )
        assert not server.store.has_session("hf-sync-session")

    finally:
        client.close_sync()
        server.stop_thread()


def test_huggingface_disaggregated_model_async_generation(
    llama_model_and_prompt: tuple[LlamaForCausalLM, torch.Tensor, list[int]],
) -> None:
    """Verify asynchronous DisaggregatedModel generates identical tokens over TCP."""
    model, prompt, expected_tokens = llama_model_and_prompt

    async def _run() -> None:
        server = ContextServer()
        port = server.start_in_thread(host="127.0.0.1", port=0)
        client = ContextClient(host="127.0.0.1", port=port)

        try:
            disagg_model = DisaggregatedModel(model=model, context_client=client)

            disagg_generated = await disagg_model.generate_async(
                input_ids=prompt,
                max_new_tokens=len(expected_tokens),
                session_id="hf-async-session",
            )

            assert expected_tokens == disagg_generated, (
                f"Generated token mismatch!\n"
                f"Expected:      {expected_tokens}\n"
                f"Disaggregated: {disagg_generated}"
            )
            assert not server.store.has_session("hf-async-session")

        finally:
            client.close_sync()
            await client.close()
            server.stop_thread()

    asyncio.run(_run())


def test_huggingface_disaggregated_model_prefix_cache_hit(
    llama_model_and_prompt: tuple[LlamaForCausalLM, torch.Tensor, list[int]],
) -> None:
    """Verify that a warm prefix cache hit bypasses prefill and produces identical tokens."""
    model, prompt, expected_tokens = llama_model_and_prompt

    server = ContextServer()
    port = server.start_in_thread(host="127.0.0.1", port=0)
    client = ContextClient(host="127.0.0.1", port=port)

    try:
        disagg_model = DisaggregatedModel(model=model, context_client=client)

        # Request 1: Cold start (runs full prefill and indexes prompt into RadixPrefixCache)
        cold_tokens = disagg_model.generate(
            input_ids=prompt,
            max_new_tokens=len(expected_tokens),
            session_id="query-1",
        )
        assert cold_tokens == expected_tokens

        # Verify that prompt is now cached in ContextServer's Radix tree
        tokens_list = prompt[0].tolist()
        m_len, _ = client.match_prefix_sync(tokens_list)
        assert m_len == len(tokens_list)

        # Request 2: Warm hit with exact same prompt (bypasses prefill!)
        warm_tokens = disagg_model.generate(
            input_ids=prompt,
            max_new_tokens=len(expected_tokens),
            session_id="query-2",
        )
        assert warm_tokens == expected_tokens

    finally:
        client.close_sync()
        server.stop_thread()


def test_mistral_sliding_window_attention_equivalence() -> None:
    """Verify MistralForCausalLM with sliding_window produces identical tokens to native HF."""
    from transformers import MistralConfig, MistralForCausalLM

    torch.manual_seed(9999)
    cfg = MistralConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        sliding_window=3,
    )
    model = MistralForCausalLM(cfg).eval()
    prompt = torch.tensor([[5, 12, 45, 99, 21]])
    num_tokens = 5

    with torch.no_grad():
        out_prefill = model(prompt, use_cache=True)
        mono_cache = copy.deepcopy(out_prefill.past_key_values)
        first_tok = out_prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        expected = [first_tok.item()]
        cur = first_tok
        for _ in range(num_tokens - 1):
            o = model(cur, past_key_values=mono_cache, use_cache=True)
            cur = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            expected.append(cur.item())

    server = ContextServer()
    port = server.start_in_thread(host="127.0.0.1", port=0)
    client = ContextClient(host="127.0.0.1", port=port)

    try:
        disagg = DisaggregatedModel(model=model, context_client=client)
        # 1. Cold prefill + sliding window decode
        actual_cold = disagg.generate(
            input_ids=prompt,
            max_new_tokens=num_tokens,
            session_id="mistral-sw-cold",
        )
        assert actual_cold == expected

        # 2. 100% prefix cache hit + sliding window decode
        actual_warm = disagg.generate(
            input_ids=prompt,
            max_new_tokens=num_tokens,
            session_id="mistral-sw-warm",
        )
        assert actual_warm == expected
    finally:
        client.close_sync()
        server.stop_thread()


def test_eos_early_termination_and_first_token_stream_cleanup(
    llama_model_and_prompt: tuple[LlamaForCausalLM, torch.Tensor, list[int]],
) -> None:
    """Verify stopping on EOS token and immediate cleanup when closing stream after first token."""
    base_model, prompt, expected_tokens = llama_model_and_prompt
    model = copy.deepcopy(base_model)

    server = ContextServer()
    port = server.start_in_thread(host="127.0.0.1", port=0)
    client = ContextClient(host="127.0.0.1", port=port)

    try:
        disagg = DisaggregatedModel(model=model, context_client=client)

        # 1. Configure first emitted token as EOS -> should stop after 1 token with finish_reason="stop"
        model.config.eos_token_id = expected_tokens[0]
        eos_out = disagg.generate(
            input_ids=prompt,
            max_new_tokens=4,
            session_id="eos-sess",
        )
        assert eos_out == [expected_tokens[0]]
        assert disagg.get_finish_reason() == "stop"

        # 2. Reset EOS and close stream after consuming only the first token -> must clean up session & patchers
        model.config.eos_token_id = -1
        stream = disagg.generate_stream(
            input_ids=prompt,
            max_new_tokens=4,
            session_id="early-close-sess",
            use_prefix_cache=False,
        )
        first = next(stream)
        assert first == expected_tokens[0]
        stream.close()

        assert not server.store.has_session("early-close-sess")
        assert all(p.active_session_id is None for p in disagg._patchers)
    finally:
        client.close_sync()
        server.stop_thread()


def test_reusing_retained_session_does_not_corrupt_cached_prompt(
    llama_model_and_prompt: tuple[LlamaForCausalLM, torch.Tensor, list[int]],
) -> None:
    """Verify reusing a retained session for a different same-length prompt does not corrupt the first prompt's cache."""
    base_model, prompt_a, expected_a = llama_model_and_prompt
    model = copy.deepcopy(base_model)
    prompt_b = torch.tensor(
        [[7, 14, 21, 28, 35]],
        dtype=torch.long,
    )
    assert prompt_a.shape == prompt_b.shape
    num_tokens = len(expected_a)

    server = ContextServer()
    port = server.start_in_thread(host="127.0.0.1", port=0)
    client = ContextClient(host="127.0.0.1", port=port)

    try:
        disagg = DisaggregatedModel(model=model, context_client=client)

        # 1. Run original prompt_a and retain session "reused-sess"
        out_a1 = disagg.generate(
            input_ids=prompt_a,
            max_new_tokens=num_tokens,
            session_id="reused-sess",
            release_on_completion=False,
        )
        assert out_a1 == expected_a

        # 2. Reuse "reused-sess" for same-length prompt_b
        out_b = disagg.generate(
            input_ids=prompt_b,
            max_new_tokens=num_tokens,
            session_id="reused-sess",
            release_on_completion=False,
        )
        assert len(out_b) == num_tokens

        # 3. Fresh cache hit on original prompt_a must still produce exact expected_a
        out_a2 = disagg.generate(
            input_ids=prompt_a,
            max_new_tokens=num_tokens,
            session_id="fresh-sess",
            use_prefix_cache=True,
        )
        assert out_a2 == expected_a
    finally:
        client.close_sync()
        server.stop_thread()


def test_disaggregated_model_temperature_and_top_p_sampling(
    llama_model_and_prompt: tuple[LlamaForCausalLM, torch.Tensor, list[int]],
) -> None:
    """Verify temperature and top_p sampling controls and validation."""
    base_model, prompt, expected_greedy = llama_model_and_prompt
    model = copy.deepcopy(base_model)
    num_tokens = len(expected_greedy)

    server = ContextServer()
    port = server.start_in_thread(host="127.0.0.1", port=0)
    client = ContextClient(host="127.0.0.1", port=port)

    try:
        disagg = DisaggregatedModel(model=model, context_client=client)

        # Very small top_p (1e-6) restricts nucleus to top-1 token even with temperature > 0
        out_top1 = disagg.generate(
            input_ids=prompt,
            max_new_tokens=num_tokens,
            session_id="top1-sess",
            temperature=0.8,
            top_p=1e-6,
        )
        assert out_top1 == expected_greedy

        with pytest.raises(ValueError, match="temperature"):
            disagg.generate(input_ids=prompt, max_new_tokens=2, temperature=-0.1)

        with pytest.raises(ValueError, match="top_p"):
            disagg.generate(input_ids=prompt, max_new_tokens=2, top_p=0.0)
    finally:
        client.close_sync()
        server.stop_thread()
