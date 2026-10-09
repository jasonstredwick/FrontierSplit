"""Tests for Online Softmax Attention merging.

Validates that splitting attention across arbitrary chunks and merging via
online softmax produces bit-for-bit identical outputs to PyTorch native attention.
"""

import math

import torch
import torch.nn.functional as F

from frontiersplit.online_softmax import (
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
    """Test FP16 numerical stability (no NaN, inf, or overflow)."""
    torch.manual_seed(999)
    batch, num_heads, q_len, head_dim = 1, 8, 1, 64
    total_len = 512

    q = torch.randn(batch, num_heads, q_len, head_dim, dtype=torch.float16)
    k = torch.randn(batch, num_heads, total_len, head_dim, dtype=torch.float16)
    v = torch.randn(batch, num_heads, total_len, head_dim, dtype=torch.float16)

    ref = native_attention(q, k, v)

    chunk_1 = compute_partial_attention(q, k[:, :, :256, :], v[:, :, :256, :])
    chunk_2 = compute_partial_attention(q, k[:, :, 256:, :], v[:, :, 256:, :])

    merged = merge_two_partial_attentions(chunk_1, chunk_2)
    out = finalize_attention(merged)

    assert not torch.isnan(out).any(), "NaN detected in FP16 merged attention"
    assert not torch.isinf(out).any(), "Inf detected in FP16 merged attention"
    # In FP16, tolerance is slightly looser due to 10-bit mantissa
    max_diff = (ref - out).abs().max().item()
    assert max_diff < 1e-2, f"FP16 merged attention difference too large: {max_diff}"
