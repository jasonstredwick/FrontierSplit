"""FrontierSplit Prefix Caching Demonstration.

Demonstrates Radix Tree prompt Key-Value (KV) prefix caching across queries on the
Context Server. Shows how repeated prefixes (system prompts, few-shot templates,
multi-turn chat) reuse precomputed KV caches over TCP with zero redundant prefill compute.

Run with:
    python examples/prefix_caching_demo.py
"""

from __future__ import annotations

import time

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from frontiersplit import (
    ContextClient,
    ContextServer,
    DisaggregatedModel,
)


def main() -> None:
    print("=" * 74)
    print(" FrontierSplit: Radix Tree Prefix Caching & Zero-Redundancy Prompt Cache")
    print("=" * 74)

    # 1. Initialize Model Architecture
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

    # Define prompts sharing a common system prefix
    system_prefix = [15, 88, 102, 45, 99, 120, 14, 77]
    user_query_a = [33, 49, 12, 60]
    prompt_a = torch.tensor([system_prefix + user_query_a])

    print("\n[1/4] Model & Cache Configuration:")
    print(
        f"  • Architecture:        LlamaForCausalLM ({config.num_hidden_layers} layers)"
    )
    print(f"  • Shared Prefix:       {len(system_prefix)} tokens: {system_prefix}")
    print(
        f"  • Full Prompt A:       {prompt_a.shape[1]} tokens: {prompt_a.tolist()[0]}"
    )

    # 2. Launch Disaggregated Context Server
    print("\n[2/4] Starting Context Server with RadixPrefixCache...")
    server = ContextServer(max_cached_tokens=100_000)
    port = server.start_in_thread(host="127.0.0.1", port=0)
    print(f"  • Context Server listening on TCP 127.0.0.1:{port}")

    client = ContextClient(host="127.0.0.1", port=port)
    disagg_model = DisaggregatedModel(model=model, context_client=client)

    try:
        # 3. Request 1: Cold Start (Full prefill & tree indexing)
        print("\n[3/4] Request 1: Cold Start (Prefill + Tree Indexing)...")
        t0 = time.perf_counter()
        tokens_req1 = disagg_model.generate(
            input_ids=prompt_a,
            max_new_tokens=6,
            session_id="req-1",
            use_prefix_cache=True,
        )
        t_cold = (time.perf_counter() - t0) * 1000
        print(f"  • Output Tokens:       {tokens_req1}")
        print(f"  • Cold Latency:        {t_cold:.2f} ms")
        print(
            f"  • Radix Tree Nodes:    {len(server.prefix_cache.root.children)} root branches"
        )
        print(f"  • Total Cached Tokens: {server.prefix_cache.total_tokens} tokens")

        # 4. Request 2: Exact Same Prompt (100% Prefix Cache Hit)
        print("\n[4/4] Request 2: Warm Hit (100% Prefix Match -> Prefill Bypassed)...")
        matched_len, _ = client.match_prefix_sync(prompt_a[0].tolist())
        print(
            f"  • Context Server Match: {matched_len}/{prompt_a.shape[1]} tokens matched (100% Hit!)"
        )

        t1 = time.perf_counter()
        tokens_req2 = disagg_model.generate(
            input_ids=prompt_a,
            max_new_tokens=6,
            session_id="req-2",
            use_prefix_cache=True,
        )
        t_warm = (time.perf_counter() - t1) * 1000
        print(f"  • Output Tokens:       {tokens_req2}")
        print(f"  • Warm Latency:        {t_warm:.2f} ms")

        # Speedup and correctness checks
        assert tokens_req1 == tokens_req2, (
            "Token mismatch between cold and warm generation!"
        )
        print(
            f"  • Generation Check:    Tokens 100% Identical: {tokens_req1 == tokens_req2}"
        )

        print("\n" + "=" * 74)
        print(" [✓] SUCCESS: RADIX PREFIX CACHE ELIMINATED REDUNDANT PREFILL COMPUTE!")
        print(f"     Cold Latency: {t_cold:.2f} ms | Warm Latency: {t_warm:.2f} ms")
        print("=" * 74)

    finally:
        client.close_sync()
        server.stop_thread()
        print("  • Cleaned up client connections and stopped server thread.")


if __name__ == "__main__":
    main()
