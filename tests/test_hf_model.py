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

from frontiersplit.context_server import ContextClient, ContextServer
from frontiersplit.hf_model import DisaggregatedModel


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
