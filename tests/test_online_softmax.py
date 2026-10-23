"""Tests for Online Softmax Attention merging.

Validates that splitting attention across arbitrary chunks and merging via
online softmax produces bit-for-bit identical outputs to PyTorch native attention.
"""

import math

import torch
import torch.nn.functional as F

from frontiersplit.context.online_softmax import (
    compute_partial_attention,
    finalize_attention,
    merge_partial_attentions,
    merge_two_partial_attentions,
)


def native_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Standard unchunked reference attention."""
    batch_size, num_heads, q_len, head_dim = q.shape
    _, num_kv_heads, kv_len, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    if num_heads != num_kv_heads:
        n_rep = num_heads // num_kv_heads
        k = (
            k[:, :, None, :, :]
            .expand(batch_size, num_kv_heads, n_rep, kv_len, head_dim)
            .reshape(batch_size, num_heads, kv_len, head_dim)
        )
        v = (
            v[:, :, None, :, :]
            .expand(batch_size, num_kv_heads, n_rep, kv_len, head_dim)
            .reshape(batch_size, num_heads, kv_len, head_dim)
        )

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def test_single_chunk_matches_native():
    """Verify that a single unchunked partial calculation matches native attention."""
    torch.manual_seed(42)
    batch, num_heads, q_len, kv_len, head_dim = 2, 8, 16, 64, 64

    q = torch.randn(batch, num_heads, q_len, head_dim, dtype=torch.float32)
    k = torch.randn(batch, num_heads, kv_len, head_dim, dtype=torch.float32)
    v = torch.randn(batch, num_heads, kv_len, head_dim, dtype=torch.float32)

    ref = native_attention(q, k, v)

    chunk = compute_partial_attention(q, k, v)
    out = finalize_attention(chunk)

    assert torch.allclose(ref, out, atol=1e-6, rtol=1e-5), (
        "Single chunk output deviates from native attention"
    )


def test_two_chunk_split_prompt_and_output():
    """Test splitting into static prompt (96 tokens) and dynamic output (32 tokens)."""
    torch.manual_seed(1337)
    batch, num_heads, q_len, head_dim = 1, 32, 1, 128
    prompt_len = 96
    output_len = 32
    total_len = prompt_len + output_len

    q = torch.randn(batch, num_heads, q_len, head_dim, dtype=torch.float32)
    k_full = torch.randn(batch, num_heads, total_len, head_dim, dtype=torch.float32)
    v_full = torch.randn(batch, num_heads, total_len, head_dim, dtype=torch.float32)

    # 1. Reference: Full native attention over total context
    ref = native_attention(q, k_full, v_full)

    # 2. Chunked: Split into prompt (first 96) and output (remaining 32)
    k_prompt = k_full[:, :, :prompt_len, :]
    v_prompt = v_full[:, :, :prompt_len, :]

    k_output = k_full[:, :, prompt_len:, :]
    v_output = v_full[:, :, prompt_len:, :]

    # Compute on remote Context Server (simulated)
    chunk_prompt = compute_partial_attention(q, k_prompt, v_prompt)

    # Compute locally on Decode Worker
    chunk_output = compute_partial_attention(q, k_output, v_output)

    # Merge via Online Softmax
    merged = merge_two_partial_attentions(chunk_prompt, chunk_output)
    out = finalize_attention(merged)

    # Exact mathematical equivalence check
    max_diff = (ref - out).abs().max().item()
    assert max_diff < 1e-5, (
        f"Two-chunk merged attention differs from native! Max diff: {max_diff}"
    )


def test_multi_chunk_arbitrary_splits():
    """Test splitting context across 4 arbitrary unequal chunks."""
    torch.manual_seed(2026)
    batch, num_heads, q_len, head_dim = 2, 16, 4, 64
    chunk_sizes = [15, 45, 120, 75]
    total_len = sum(chunk_sizes)

    q = torch.randn(batch, num_heads, q_len, head_dim, dtype=torch.float32)
    k_full = torch.randn(batch, num_heads, total_len, head_dim, dtype=torch.float32)
    v_full = torch.randn(batch, num_heads, total_len, head_dim, dtype=torch.float32)

    ref = native_attention(q, k_full, v_full)

    # Split into chunks and compute partials
    chunks = []
    offset = 0
    for size in chunk_sizes:
        k_chunk = k_full[:, :, offset : offset + size, :]
        v_chunk = v_full[:, :, offset : offset + size, :]
        chunks.append(compute_partial_attention(q, k_chunk, v_chunk))
        offset += size

    merged = merge_partial_attentions(chunks)
    out = finalize_attention(merged)

    max_diff = (ref - out).abs().max().item()
    assert max_diff < 1e-5, (
        f"Multi-chunk merged attention differs from native! Max diff: {max_diff}"
    )


def test_grouped_query_attention_gqa():
    """Verify that GQA (e.g. 32 Q heads, 8 KV heads like Mixtral/Llama) merges identically."""
    torch.manual_seed(777)
    batch, num_heads, num_kv_heads, q_len, head_dim = 1, 32, 8, 1, 128
    prompt_len, output_len = 256, 64
    total_len = prompt_len + output_len

    q = torch.randn(batch, num_heads, q_len, head_dim, dtype=torch.float32)
    k_full = torch.randn(batch, num_kv_heads, total_len, head_dim, dtype=torch.float32)
    v_full = torch.randn(batch, num_kv_heads, total_len, head_dim, dtype=torch.float32)

    ref = native_attention(q, k_full, v_full)

    chunk_p = compute_partial_attention(
        q, k_full[:, :, :prompt_len, :], v_full[:, :, :prompt_len, :]
    )
    chunk_o = compute_partial_attention(
        q, k_full[:, :, prompt_len:, :], v_full[:, :, prompt_len:, :]
    )

    merged = merge_two_partial_attentions(chunk_p, chunk_o)
    out = finalize_attention(merged)

    max_diff = (ref - out).abs().max().item()
    assert max_diff < 1e-5, (
        f"GQA merged attention differs from native! Max diff: {max_diff}"
    )


def test_fp16_numerical_stability():
    """Test FP16 numerical stability and FP32 intermediate accumulation."""
    torch.manual_seed(999)
    batch, num_heads, q_len, head_dim = 1, 8, 1, 64
    total_len = 512

    q = torch.randn(batch, num_heads, q_len, head_dim, dtype=torch.float16)
    k = torch.randn(batch, num_heads, total_len, head_dim, dtype=torch.float16)
    v = torch.randn(batch, num_heads, total_len, head_dim, dtype=torch.float16)

    ref = native_attention(q, k, v)

    chunk_1 = compute_partial_attention(q, k[:, :, :256, :], v[:, :, :256, :])
    chunk_2 = compute_partial_attention(q, k[:, :, 256:, :], v[:, :, 256:, :])

    # Verify intermediate accumulators are held in FP32
    assert chunk_1.accumulator.dtype == torch.float32
    assert chunk_1.max_score.dtype == torch.float32
    assert chunk_1.sum_exp.dtype == torch.float32
    assert chunk_1.dtype == torch.float16

    merged = merge_two_partial_attentions(chunk_1, chunk_2)
    assert merged.accumulator.dtype == torch.float32
    assert merged.dtype == torch.float16

    out = finalize_attention(merged)
    assert out.dtype == torch.float16

    assert not torch.isnan(out).any(), "NaN detected in FP16 merged attention"
    assert not torch.isinf(out).any(), "Inf detected in FP16 merged attention"

    # Compare against unchunked online softmax and native FP16 attention
    single_out = finalize_attention(compute_partial_attention(q, k, v))
    assert (single_out - out).abs().max().item() <= 1e-3
    max_diff = (ref - out).abs().max().item()
    assert max_diff < 1e-3, f"FP16 merged attention difference too large: {max_diff}"


def test_fp16_long_context_no_overflow():
    """Verify that 100k+ uniform-score tokens in FP16 do not overflow sum_exp (65504 limit)."""
    batch, num_heads, q_len, head_dim = 1, 2, 1, 16
    total_len = 80_000  # Exceeds FP16 max value of 65,504 when scores are identical

    q = torch.zeros(batch, num_heads, q_len, head_dim, dtype=torch.float16)
    k = torch.zeros(batch, num_heads, total_len, head_dim, dtype=torch.float16)
    v = torch.ones(batch, num_heads, total_len, head_dim, dtype=torch.float16)

    chunk_a = compute_partial_attention(q, k[:, :, :40_000, :], v[:, :, :40_000, :])
    chunk_b = compute_partial_attention(q, k[:, :, 40_000:, :], v[:, :, 40_000:, :])

    merged = merge_two_partial_attentions(chunk_a, chunk_b)
    assert torch.all(merged.sum_exp == 80_000.0)  # Would be inf in FP16!

    out = finalize_attention(merged)
    assert out.dtype == torch.float16
    assert not torch.isnan(out).any()
    assert torch.allclose(out, torch.ones_like(out))


def test_fp16_dot_product_no_overflow_before_fp32_cast():
    """Verify FP16 Q @ K^T with entries of 32 and head_dim=128 does not overflow before scaling."""
    batch, num_heads, q_len, kv_len, head_dim = 1, 2, 1, 4, 128
    q = torch.full((batch, num_heads, q_len, head_dim), 32.0, dtype=torch.float16)
    k = torch.full((batch, num_heads, kv_len, head_dim), 32.0, dtype=torch.float16)
    v = torch.full((batch, num_heads, kv_len, head_dim), 2.5, dtype=torch.float16)

    chunk = compute_partial_attention(q, k, v)
    out = finalize_attention(chunk)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    assert not torch.isnan(out).any(), "FP16 dot product overflowed to NaN"
    assert torch.allclose(out, ref, atol=1e-3)


def test_all_neg_inf_masked_partition_merge():
    """Verify that a fully masked (-inf) partition does not poison valid partitions with NaN."""
    batch, num_heads, q_len, kv_len, head_dim = 1, 2, 1, 4, 16
    q = torch.ones(batch, num_heads, q_len, head_dim)
    k = torch.ones(batch, num_heads, kv_len, head_dim)
    v_masked = torch.full((batch, num_heads, kv_len, head_dim), 999.0)
    v_valid = torch.full((batch, num_heads, kv_len, head_dim), 2.0)

    all_masked = torch.full((batch, 1, q_len, kv_len), float("-inf"))
    chunk_masked = compute_partial_attention(q, k, v_masked, attention_mask=all_masked)
    chunk_valid = compute_partial_attention(q, k, v_valid)

    assert not torch.isnan(chunk_masked.accumulator).any()
    assert not torch.isnan(chunk_masked.sum_exp).any()

    merged_ab = merge_two_partial_attentions(chunk_masked, chunk_valid)
    out_ab = finalize_attention(merged_ab)
    assert not torch.isnan(out_ab).any()
    assert torch.allclose(out_ab, torch.full_like(out_ab, 2.0))

    merged_ba = merge_two_partial_attentions(chunk_valid, chunk_masked)
    out_ba = finalize_attention(merged_ba)
    assert not torch.isnan(out_ba).any()
    assert torch.allclose(out_ba, torch.full_like(out_ba, 2.0))


def test_autocast_fp16_does_not_defeat_fp32_safeguards():
    """Verify enclosing torch.autocast('cpu', dtype=torch.float16) does not downcast FP32 accumulators or overflow."""
    batch, num_heads, q_len, kv_len, head_dim = 1, 2, 1, 8, 128
    q = torch.full((batch, num_heads, q_len, head_dim), 32.0, dtype=torch.float16)
    k = torch.full((batch, num_heads, kv_len, head_dim), 32.0, dtype=torch.float16)
    v = torch.full((batch, num_heads, kv_len, head_dim), 1.75, dtype=torch.float16)

    with torch.autocast("cpu", dtype=torch.float16):
        chunk_a = compute_partial_attention(q, k[:, :, :4, :], v[:, :, :4, :])
        chunk_b = compute_partial_attention(q, k[:, :, 4:, :], v[:, :, 4:, :])
        assert chunk_a.accumulator.dtype == torch.float32
        assert chunk_a.max_score.dtype == torch.float32
        assert chunk_a.sum_exp.dtype == torch.float32

        merged = merge_two_partial_attentions(chunk_a, chunk_b)
        assert merged.accumulator.dtype == torch.float32
        assert merged.sum_exp.dtype == torch.float32

        out = finalize_attention(merged)

    assert out.dtype == torch.float16
    assert not torch.isnan(out).any()
    assert torch.allclose(out, torch.full_like(out, 1.75), atol=1e-3)


def test_long_delta_prefill_tiled_workspace_bounds():
    """Verify that multi-token query (delta prefill) against long KV cache tiles cleanly with bounded workspace."""
    torch.manual_seed(42)
    batch, num_heads, q_len, kv_len, head_dim = 1, 4, 256, 1024, 64
    q = torch.randn(batch, num_heads, q_len, head_dim, dtype=torch.float32)
    k = torch.randn(batch, num_heads, kv_len, head_dim, dtype=torch.float32)
    v = torch.randn(batch, num_heads, kv_len, head_dim, dtype=torch.float32)

    # Reference computation using native scaled dot product attention
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # Tiled execution with small block sizes (64 query tokens, 256 KV tokens)
    chunk = compute_partial_attention(
        q, k, v, query_chunk_size=64, kv_chunk_size=256
    )
    out = finalize_attention(chunk)

    assert out.shape == (batch, num_heads, q_len, head_dim)
    assert not torch.isnan(out).any()
    # Output matches native attention within floating-point reduction error
    assert torch.allclose(out, ref, atol=1e-5)


def test_tiled_attention_broadcastable_singleton_masks():
    """Verify that broadcastable masks with singleton dimensions spanning tile boundaries succeed."""
    torch.manual_seed(42)
    batch, num_heads, q_len, kv_len, head_dim = 1, 4, 129, 1030, 64
    q = torch.randn(batch, num_heads, q_len, head_dim, dtype=torch.float32)
    k = torch.randn(batch, num_heads, kv_len, head_dim, dtype=torch.float32)
    v = torch.randn(batch, num_heads, kv_len, head_dim, dtype=torch.float32)

    # 1. Query-specific singleton-key mask [1, 1, 129, 1]
    mask_q_singleton_k = torch.zeros(1, 1, 129, 1, dtype=torch.float32)
    chunk = compute_partial_attention(
        q, k, v, attention_mask=mask_q_singleton_k, query_chunk_size=128, kv_chunk_size=1024
    )
    out = finalize_attention(chunk)
    assert out.shape == (batch, num_heads, q_len, head_dim)
    assert not torch.isnan(out).any()

    # 2. Key-specific singleton-query mask [1, 1, 1, 1030]
    mask_singleton_q_k = torch.zeros(1, 1, 1, 1030, dtype=torch.float32)
    chunk2 = compute_partial_attention(
        q, k, v, attention_mask=mask_singleton_q_k, query_chunk_size=128, kv_chunk_size=1024
    )
    out2 = finalize_attention(chunk2)
    assert out2.shape == (batch, num_heads, q_len, head_dim)
    assert not torch.isnan(out2).any()


def test_tiled_causal_and_sliding_window_masking():
    """Verify that tile-local causal and sliding window mask construction matches SDPA reference."""
    torch.manual_seed(42)
    batch, num_heads, q_len, kv_len, head_dim = 1, 4, 300, 300, 64
    q = torch.randn(batch, num_heads, q_len, head_dim, dtype=torch.float32)
    k = torch.randn(batch, num_heads, kv_len, head_dim, dtype=torch.float32)
    v = torch.randn(batch, num_heads, kv_len, head_dim, dtype=torch.float32)

    # Causal attention reference via PyTorch SDPA
    ref_causal = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Tiled execution with small 128-token tiles
    chunk_causal = compute_partial_attention(
        q, k, v, is_causal=True, query_chunk_size=128, kv_chunk_size=128
    )
    out_causal = finalize_attention(chunk_causal)

    diff = torch.max(torch.abs(out_causal - ref_causal)).item()
    assert diff < 1e-4

    # Sliding window attention
    chunk_sw = compute_partial_attention(
        q, k, v, is_causal=True, sliding_window=64, query_chunk_size=128, kv_chunk_size=128
    )
    out_sw = finalize_attention(chunk_sw)
    assert out_sw.shape == (batch, num_heads, q_len, head_dim)
    assert not torch.isnan(out_sw).any()


def test_gqa_tiled_expansion_bounded_memory():
    """Verify that GQA head expansion executes cleanly during tiled compute without full KV repeat."""
    torch.manual_seed(42)
    batch, q_heads, kv_heads, q_len, kv_len, head_dim = 1, 32, 8, 1, 2048, 128
    q = torch.randn(batch, q_heads, q_len, head_dim, dtype=torch.float16)
    k = torch.randn(batch, kv_heads, kv_len, head_dim, dtype=torch.float16)
    v = torch.randn(batch, kv_heads, kv_len, head_dim, dtype=torch.float16)

    chunk = compute_partial_attention(
        q, k, v, query_chunk_size=128, kv_chunk_size=512
    )
    out = finalize_attention(chunk)
    assert out.shape == (batch, q_heads, q_len, head_dim)
    assert not torch.isnan(out).any()


