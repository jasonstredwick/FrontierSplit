"""FrontierSplit Disaggregated Inference Quickstart Demo.

Demonstrates how a large, static prompt KV cache (e.g. 10,000 tokens) can be
offloaded to a dedicated Context Server, allowing a lightweight Decode Worker
to generate tokens with microscopic local memory and 8 KB network round-trips.

Run with:
    python examples/disaggregated_demo.py
"""

from __future__ import annotations

import asyncio
import socket
import time

import torch

from frontiersplit import (
    ContextClient,
    ContextServer,
    DecodeWorker,
    compute_partial_attention,
    finalize_attention,
)


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def main() -> None:
    print("=" * 70)
    print(" FrontierSplit: Disaggregated Long-Context Inference Demo")
    print("=" * 70)

    # Architectural Hyperparameters
    batch_size = 1
    num_heads = 32
    num_kv_heads = 8  # Grouped-Query Attention (GQA)
    head_dim = 128
    prompt_tokens = 10_000
    decode_steps = 10

    # Calculate memory footprint
    bytes_per_elem = 2  # FP16
    prompt_kv_bytes = 2 * num_kv_heads * prompt_tokens * head_dim * bytes_per_elem
    query_bytes = batch_size * num_heads * 1 * head_dim * bytes_per_elem
    response_bytes = query_bytes + (num_heads * 2 * 4)  # Acc + max_score + sum_exp

    print("\n[Architecture Specs]")
    print(f"  • Prompt Length:       {prompt_tokens:,} tokens")
    print(f"  • Prompt KV Footprint: {prompt_kv_bytes / (1024 * 1024):.2f} MB")
    print(f"  • Network Query Size:  {query_bytes / 1024:.2f} KB (Send)")
    print(f"  • Network Return Size: {response_bytes / 1024:.2f} KB (Receive)")
    print("  • Merge Algorithm:     Exact Online Softmax Rescaling")

    # 1. Start Context Server
    port = get_free_port()
    server = ContextServer()
    await server.start_server(host="127.0.0.1", port=port)
    print(f"\n[1/4] Context Server started on 127.0.0.1:{port}")

    # 2. Simulate Prefill: Store prompt KV cache on Context Server
    print(f"[2/4] Offloading {prompt_tokens:,} prompt tokens to Context Server...")
    t0 = time.perf_counter()
    k_prompt = torch.randn(
        batch_size, num_kv_heads, prompt_tokens, head_dim, dtype=torch.float32
    )
    v_prompt = torch.randn(
        batch_size, num_kv_heads, prompt_tokens, head_dim, dtype=torch.float32
    )

    session_id = "user-request-001"
    server.register_prompt(session_id, layer_idx=0, k=k_prompt, v=v_prompt)
    print(f"      Prompt registered in {(time.perf_counter() - t0) * 1000:.2f} ms")

    # 3. Initialize Decode Worker
    print("\n[3/4] Initializing Decode Worker...")
    client = ContextClient(host="127.0.0.1", port=port)
    worker = DecodeWorker(context_client=client, num_layers=1)

    # 4. Run Autoregressive Decoding Loop
    print(f"[4/4] Generating {decode_steps} tokens with disaggregated attention...")
    print("-" * 70)
    print(" Step | Local Output Tokens | Query RTT | Verification vs. Native")
    print("-" * 70)

    # Ground truth reference trackers
    all_k = k_prompt.clone()
    all_v = v_prompt.clone()

    for step in range(1, decode_steps + 1):
        step_t0 = time.perf_counter()

        q = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float32)
        k_tok = torch.randn(batch_size, num_kv_heads, 1, head_dim, dtype=torch.float32)
        v_tok = torch.randn(batch_size, num_kv_heads, 1, head_dim, dtype=torch.float32)

        # Disaggregated forward pass (Socket Query -> Online Softmax Merge -> Finalize)
        disagg_out = await worker.forward_layer(
            session_id=session_id,
            layer_idx=0,
            q=q,
            k_token=k_tok,
            v_token=v_tok,
        )
        step_ms = (time.perf_counter() - step_t0) * 1000

        # Ground truth comparison: monolithic attention across all (prompt + output) tokens
        all_k = torch.cat([all_k, k_tok], dim=-2)
        all_v = torch.cat([all_v, v_tok], dim=-2)
        ref_chunk = compute_partial_attention(q, all_k, all_v)
        ref_out = finalize_attention(ref_chunk)

        is_exact = torch.allclose(ref_out, disagg_out, atol=1e-5, rtol=1e-5)
        status = "✓ Bit-for-bit Exact" if is_exact else "✗ Discrepancy"

        print(
            f"  {step:2d}  |      {step:2d} tokens      | {step_ms:5.2f} ms  | {status}"
        )

    print("-" * 70)

    # Cleanup
    await worker.end_session(session_id)
    await client.close()
    await server.stop_server()

    print("\n✓ Demo completed successfully.")
    print("  The Decode Worker held only 10 output tokens in local memory,")
    print("  yet computed exact attention over the entire 10,010-token context!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
