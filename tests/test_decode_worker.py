"""Unit tests for the Disaggregated Decode Worker.

Simulates an end-to-end multi-step autoregressive decoding sequence with
disaggregated attention and validates bit-for-bit equivalence against
native unchunked attention at every step.
"""

from __future__ import annotations

import asyncio
import socket

import torch

from frontiersplit.context_server import ContextClient, ContextServer
from frontiersplit.decode_worker import DecodeWorker, LocalOutputKVCache
from frontiersplit.online_softmax import compute_partial_attention, finalize_attention


def get_free_port() -> int:
    """Finds an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_local_output_kv_cache_growth():
    """Verify that LocalOutputKVCache appends tokens correctly."""
    cache = LocalOutputKVCache()
    batch, num_kv_heads, head_dim = 1, 4, 32

    t0_k = torch.randn(batch, num_kv_heads, 1, head_dim)
    t0_v = torch.randn(batch, num_kv_heads, 1, head_dim)

    k_out, v_out = cache.append("s1", layer_idx=0, k_token=t0_k, v_token=t0_v)
    assert k_out.shape == (batch, num_kv_heads, 1, head_dim)
    assert v_out.shape == (batch, num_kv_heads, 1, head_dim)

    t1_k = torch.randn(batch, num_kv_heads, 1, head_dim)
    t1_v = torch.randn(batch, num_kv_heads, 1, head_dim)

    k_out, v_out = cache.append("s1", layer_idx=0, k_token=t1_k, v_token=t1_v)
    assert k_out.shape == (batch, num_kv_heads, 2, head_dim)
    assert v_out.shape == (batch, num_kv_heads, 2, head_dim)

    # Release
    cache.release_session("s1")
    assert cache.get("s1", layer_idx=0) is None


def test_autoregressive_decode_step_by_step_simulation():
    """Simulates a multi-step decode run and verifies exact attention at every step."""

    async def _run() -> None:
        port = get_free_port()
        server = ContextServer()
        await server.start_server(host="127.0.0.1", port=port)

        client = ContextClient(host="127.0.0.1", port=port)
        worker = DecodeWorker(context_client=client, num_layers=2)

        try:
            batch, num_heads, num_kv_heads, head_dim = 1, 8, 4, 64
            prompt_len = 64
            decode_steps = 10

            # 1. Generate static prompt KV
            k_prompt = torch.randn(batch, num_kv_heads, prompt_len, head_dim)
            v_prompt = torch.randn(batch, num_kv_heads, prompt_len, head_dim)

            session_id = "test-stream-1"
            await client.register_prompt(
                session_id=session_id,
                layer_idx=0,
                k=k_prompt,
                v=v_prompt,
            )

            # Ground-truth accumulators for verification
            all_k = k_prompt.clone()
            all_v = v_prompt.clone()

            # 2. Simulate 10 sequential decode steps
            for step in range(decode_steps):
                q_step = torch.randn(batch, num_heads, 1, head_dim)
                k_step = torch.randn(batch, num_kv_heads, 1, head_dim)
                v_step = torch.randn(batch, num_kv_heads, 1, head_dim)

                # Ground truth: append to full sequence and run native attention
                all_k = torch.cat([all_k, k_step], dim=-2)
                all_v = torch.cat([all_v, v_step], dim=-2)

                ref_chunk = compute_partial_attention(q_step, all_k, all_v)
                expected_out = finalize_attention(ref_chunk)

                # Disaggregated Decode Worker step
                actual_out = await worker.forward_layer(
                    session_id=session_id,
                    layer_idx=0,
                    q=q_step,
                    k_token=k_step,
                    v_token=v_step,
                )

                # Bit-for-bit equivalence check at this step
                assert torch.allclose(expected_out, actual_out, atol=1e-5, rtol=1e-5), (
                    f"Mismatch at decode step {step}!"
                )

            # 3. End session and verify complete cleanup
            await worker.end_session(session_id)
            assert not worker.kv_cache.get(session_id, layer_idx=0)
            assert not server.store.has_session(session_id)

        finally:
            await client.close()
            await server.stop_server()

    asyncio.run(_run())
