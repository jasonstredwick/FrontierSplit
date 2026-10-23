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


def test_disaggregated_model_reasoning_commit_threshold(
    llama_model_and_prompt: tuple[LlamaForCausalLM, torch.Tensor, list[int]],
) -> None:
    """Verify reasoning commit policy (Review 4.3): outputs >= threshold are committed to Context Server."""
    base_model, prompt_t1, _ = llama_model_and_prompt
    model = copy.deepcopy(base_model)
    server = ContextServer()
    port = server.start_in_thread(host="127.0.0.1", port=0)
    client = ContextClient(host="127.0.0.1", port=port)

    session_id = "reasoning-commit-session"
    try:
        disagg = DisaggregatedModel(model=model, context_client=client)

        # Turn 1: generate 5 tokens with reasoning_commit_threshold=4 and release_on_completion=False
        out_t1 = disagg.generate(
            input_ids=prompt_t1,
            max_new_tokens=5,
            session_id=session_id,
            release_on_completion=False,
            reasoning_commit_threshold=4,
        )
        assert len(out_t1) == 5

        # Verify ContextStore now holds the prompt + reasoning output tokens
        prompt_len = prompt_t1.shape[1]
        kv_committed = server.store.get_prompt(session_id, 0)
        assert kv_committed is not None
        assert kv_committed[0].shape[-2] >= prompt_len + 4

        # Turn 2: Construct continuation prompt including the committed reasoning tokens + new user tokens
        committed_reasoning_tokens = out_t1[:4]
        new_user_tokens = [99, 100]
        prompt_t2 = torch.cat(
            [
                prompt_t1,
                torch.tensor(
                    [committed_reasoning_tokens + new_user_tokens], dtype=torch.long
                ),
            ],
            dim=1,
        )

        # Turn 2 match_prefix on Context Server must match all prompt + reasoning tokens
        matched_len, _ = client.match_prefix_sync(
            prompt_t2[0].tolist(), session_id=session_id
        )
        assert matched_len >= prompt_len + 4

        # Run Turn 2 generation
        out_t2 = disagg.generate(
            input_ids=prompt_t2,
            max_new_tokens=2,
            session_id=session_id,
            use_prefix_cache=True,
            release_on_completion=True,
        )
        assert len(out_t2) == 2
        # After completion with release_on_completion=True, session is released
        assert not server.store.has_session(session_id)
    finally:
        client.close_sync()
        server.stop_thread()


def test_decode_multi_commit_positional_stability(
    llama_model_and_prompt: tuple[LlamaForCausalLM, torch.Tensor, list[int]],
) -> None:
    """Verify that multiple mid-generation reasoning commits do not corrupt token positions.

    Review finding P1.2: Commits incremented prompt_len while step also counted generated tokens,
    causing positions to jump (e.g. 5, 6, 9, 10, 13, 14...) and changing the generated output.
    With the fix, generation through multiple commits (threshold=2 for 8 tokens) must match
    the uncommitted baseline exactly bit-for-bit.
    """
    base_model, prompt, _ = llama_model_and_prompt
    model = copy.deepcopy(base_model)
    server = ContextServer()
    port = server.start_in_thread(host="127.0.0.1", port=0)
    client = ContextClient(host="127.0.0.1", port=port)

    try:
        disagg = DisaggregatedModel(model=model, context_client=client)

        # Baseline: generate 8 tokens without mid-generation commits
        baseline_tokens = disagg.generate(
            input_ids=prompt,
            max_new_tokens=8,
            session_id="baseline-pos-session",
            reasoning_commit_threshold=0,
            release_on_completion=True,
        )
        assert len(baseline_tokens) == 8

        # Multi-commit run: generate 8 tokens with threshold=2 (commits at tokens 2, 4, 6, etc.)
        multi_commit_tokens = disagg.generate(
            input_ids=prompt,
            max_new_tokens=8,
            session_id="multicommit-pos-session",
            reasoning_commit_threshold=2,
            release_on_completion=True,
        )
        assert len(multi_commit_tokens) == 8

        # The generated token sequence MUST be identical to the baseline
        assert multi_commit_tokens == baseline_tokens
    finally:
        client.close_sync()
        server.stop_thread()


def test_retained_session_first_token_eos_and_early_close(
    llama_model_and_prompt: tuple[LlamaForCausalLM, torch.Tensor, list[int]],
) -> None:
    """Verify that retained sessions do not raise UnboundLocalError on first-token EOS or early close.

    Review finding P1.3: generated_tokens and local_cache were defined after the first yield;
    first-token EOS or early generator closure caused UnboundLocalError in finally, leaving
    the active session set on model patchers.
    """
    base_model, prompt, _ = llama_model_and_prompt
    model = copy.deepcopy(base_model)
    server = ContextServer()
    port = server.start_in_thread(host="127.0.0.1", port=0)
    client = ContextClient(host="127.0.0.1", port=port)

    async def _run() -> None:
        try:
            disagg = DisaggregatedModel(model=model, context_client=client)

            # 1. First-token EOS: configure max_new_tokens=1 with release_on_completion=False
            out = disagg.generate(
                input_ids=prompt,
                max_new_tokens=1,
                session_id="eos-early-sess",
                release_on_completion=False,
                reasoning_commit_threshold=4,
            )
            assert len(out) == 1
            # Patchers must have active_session_id cleared
            for p in disagg._patchers:
                assert p.active_session_id is None

            # 2. Synchronous stream early close after 1 yield
            gen = disagg.generate_stream(
                input_ids=prompt,
                max_new_tokens=10,
                session_id="sync-early-close",
                release_on_completion=False,
                reasoning_commit_threshold=4,
            )
            first_tok = next(gen)
            assert isinstance(first_tok, int)
            gen.close()  # Generator closed early -> triggers finally block
            for p in disagg._patchers:
                assert p.active_session_id is None

            # 3. Asynchronous stream early close after 1 yield
            agen = disagg.generate_stream_async(
                input_ids=prompt,
                max_new_tokens=10,
                session_id="async-early-close",
                release_on_completion=False,
                reasoning_commit_threshold=4,
            )
            first_async_tok = await anext(agen)
            assert isinstance(first_async_tok, int)
            await (
                agen.aclose()
            )  # Async generator closed early -> triggers finally block
            for p in disagg._patchers:
                assert p.active_session_id is None

        finally:
            client.close_sync()
            await client.close()
            server.stop_thread()

    asyncio.run(_run())


def test_multiplexed_generators_isolated_client_bindings(
    llama_model_and_prompt: tuple[LlamaForCausalLM, torch.Tensor, list[int]],
) -> None:
    """Verify that two streams multiplexed on one thread (next(A), next(B), next(A)) do not cross-bleed clients."""
    base_model, prompt, _ = llama_model_and_prompt
    model = copy.deepcopy(base_model)

    server_a = ContextServer()
    port_a = server_a.start_in_thread(host="127.0.0.1", port=0)
    client_a = ContextClient(host="127.0.0.1", port=port_a)

    server_b = ContextServer()
    port_b = server_b.start_in_thread(host="127.0.0.1", port=0)
    client_b = ContextClient(host="127.0.0.1", port=port_b)

    disagg = DisaggregatedModel(model=model, context_client=client_a)

    try:
        # Start generator A bound to client_a
        gen_a = disagg.generate_stream(
            input_ids=prompt,
            max_new_tokens=4,
            session_id="stream-a",
            context_client=client_a,
            release_on_completion=False,
        )
        # Start generator B bound to client_b
        gen_b = disagg.generate_stream(
            input_ids=prompt,
            max_new_tokens=4,
            session_id="stream-b",
            context_client=client_b,
            release_on_completion=False,
        )

        # Interleave execution: next(A) -> next(B) -> next(A) -> next(B)
        tok_a1 = next(gen_a)
        assert isinstance(tok_a1, int)
        tok_b1 = next(gen_b)
        assert isinstance(tok_b1, int)
        tok_a2 = next(gen_a)
        assert isinstance(tok_a2, int)
        tok_b2 = next(gen_b)
        assert isinstance(tok_b2, int)

        # Verify that session-a is strictly on server_a, session-b is strictly on server_b
        assert server_a.store.has_session("stream-a")
        assert not server_a.store.has_session("stream-b")
        assert server_b.store.has_session("stream-b")
        assert not server_b.store.has_session("stream-a")

        # Explicitly close generators (triggers finally blocks)
        gen_a.close()
        gen_b.close()

    finally:
        client_a.close_sync()
        client_b.close_sync()
        server_a.stop_thread()
        server_b.stop_thread()


def test_async_generation_does_not_block_event_loop(
    llama_model_and_prompt: tuple[LlamaForCausalLM, torch.Tensor, list[int]],
) -> None:
    """Verify that model execution runs off the event loop thread, allowing asyncio heartbeats to tick."""
    base_model, prompt, _ = llama_model_and_prompt
    model = copy.deepcopy(base_model)

    server = ContextServer()
    port = server.start_in_thread(host="127.0.0.1", port=0)
    client = ContextClient(host="127.0.0.1", port=port)
    disagg = DisaggregatedModel(model=model, context_client=client)

    # Wrap the original query_partial_attention on server with a 10ms injected delay
    original_query = server.query_partial_attention

    def delayed_query(*args, **kwargs):
        import time

        time.sleep(0.01)  # 10ms artificial delay
        return original_query(*args, **kwargs)

    server.query_partial_attention = delayed_query

    async def _run() -> None:
        heartbeat_delays: list[float] = []
        stop_heartbeat = asyncio.Event()

        async def heartbeat():
            import time

            while not stop_heartbeat.is_set():
                t0 = time.monotonic()
                await asyncio.sleep(0.005)
                dt = time.monotonic() - t0
                heartbeat_delays.append(dt)

        hb_task = asyncio.create_task(heartbeat())
        try:
            tokens: list[int] = []
            async for tok in disagg.generate_stream_async(
                input_ids=prompt,
                max_new_tokens=6,
                session_id="heartbeat-sess",
                release_on_completion=True,
            ):
                tokens.append(tok)
            assert len(tokens) == 6
        finally:
            stop_heartbeat.set()
            await hb_task

        # If the event loop was blocked during the 6 decode steps (6 x 10ms = 60ms),
        # a heartbeat would be stalled for > 50ms.
        # Since model runs in asyncio.to_thread, max heartbeat delay should remain small (< 40ms).
        max_delay = max(heartbeat_delays) if heartbeat_delays else 0.0
        assert max_delay < 0.05, f"Event loop was blocked: max heartbeat delay was {max_delay * 1000:.1f}ms"

    try:
        asyncio.run(_run())
    finally:
        client.close_sync()
        server.stop_thread()

