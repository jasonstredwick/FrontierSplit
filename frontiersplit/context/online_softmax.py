"""Online Softmax Attention primitives for disaggregated inference.

This module implements exact, numerically stable online softmax chunking for
attention mechanisms (Dao et al. 2022, Milakov & Gimelshein 2018).

It allows attention over a long context to be computed across separate chunks
(e.g., a static prompt cache on a remote Context Server and a dynamic local
output cache on a Decode Worker) and merged into a mathematically exact final
attention output with zero accuracy loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "PartialAttentionChunk",
    "compute_partial_attention",
    "finalize_attention",
    "merge_partial_attentions",
    "merge_two_partial_attentions",
    "repeat_kv",
]


@dataclass
class PartialAttentionChunk:
    """Represents a partial attention calculation over a key-value chunk.

    Intermediate statistics (`accumulator`, `max_score`, `sum_exp`) are maintained
    in `torch.float32` to prevent mantissa rounding and FP16 dynamic-range overflow
    prior to final normalization.

    Attributes:
        accumulator: Unnormalized weighted value accumulator of shape
            `[batch, num_heads, q_len, head_dim]` in `torch.float32`.
        max_score: Maximum attention logit per query position of shape
            `[batch, num_heads, q_len, 1]` in `torch.float32`.
        sum_exp: Sum of exponentials (denominator) of shape
            `[batch, num_heads, q_len, 1]` in `torch.float32`.
        dtype: Target output dtype to restore upon finalization (e.g. `torch.float16`).
    """

    accumulator: torch.Tensor
    max_score: torch.Tensor
    sum_exp: torch.Tensor
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if self.dtype is None:
            self.dtype = self.accumulator.dtype

    def to(self, device: torch.device | str) -> PartialAttentionChunk:
        """Moves chunk tensors to the specified device."""
        return PartialAttentionChunk(
            accumulator=self.accumulator.to(device),
            max_score=self.max_score.to(device),
            sum_exp=self.sum_exp.to(device),
            dtype=self.dtype,
        )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expands KV heads to match query heads for Grouped-Query Attention (GQA)."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def compute_partial_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    attention_mask: torch.Tensor | None = None,
) -> PartialAttentionChunk:
    """Computes partial attention against a chunk of keys and values.

    Args:
        q: Query tensor of shape `[batch, num_heads, q_len, head_dim]`.
        k: Key tensor of shape `[batch, num_kv_heads, kv_len, head_dim]`.
        v: Value tensor of shape `[batch, num_kv_heads, kv_len, head_dim]`.
        scale: Softmax temperature scale factor (default: 1 / sqrt(head_dim)).
        attention_mask: Optional mask of shape `[batch, 1, q_len, kv_len]`.

    Returns:
        A PartialAttentionChunk containing (accumulator, max_score, sum_exp).
    """
    batch_size, num_heads, q_len, head_dim = q.shape
    _, num_kv_heads, kv_len, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Support GQA: expand KV heads if num_heads != num_kv_heads
    if num_heads != num_kv_heads:
        n_rep = num_heads // num_kv_heads
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

    orig_dtype = q.dtype

    if kv_len == 0:
        return PartialAttentionChunk(
            accumulator=torch.zeros(
                (batch_size, num_heads, q_len, head_dim),
                dtype=torch.float32,
                device=q.device,
            ),
            max_score=torch.full(
                (batch_size, num_heads, q_len, 1),
                float("-inf"),
                dtype=torch.float32,
                device=q.device,
            ),
            sum_exp=torch.zeros(
                (batch_size, num_heads, q_len, 1),
                dtype=torch.float32,
                device=q.device,
            ),
            dtype=orig_dtype,
        )

    # Upcast Q and K to FP32 prior to matmul so unscaled dot products cannot overflow FP16:
    # [batch, num_heads, q_len, kv_len]
    scores = (
        torch.matmul(q.to(torch.float32), k.transpose(-2, -1).to(torch.float32)) * scale
    )

    if attention_mask is not None:
        scores = scores + attention_mask.to(torch.float32)

    # Numerically stable online softmax tracking in FP32
    # 1. Local maximum per query: [batch, num_heads, q_len, 1]
    max_score = torch.max(scores, dim=-1, keepdim=True).values

    # Guard against (-inf) - (-inf) = NaN when an entire partition is masked out
    safe_max = torch.where(
        torch.isneginf(max_score), torch.zeros_like(max_score), max_score
    )

    # 2. Unnormalized exponentiated probabilities: [batch, num_heads, q_len, kv_len]
    p = torch.exp(scores - safe_max)

    # 3. Sum of exponentials (denominator): [batch, num_heads, q_len, 1]
    sum_exp = torch.sum(p, dim=-1, keepdim=True)

    # 4. Partial accumulator (numerator) in FP32: [batch, num_heads, q_len, head_dim]
    accumulator = torch.matmul(p, v.to(torch.float32))
    accumulator = torch.where(sum_exp == 0, torch.zeros_like(accumulator), accumulator)

    return PartialAttentionChunk(
        accumulator=accumulator,
        max_score=max_score,
        sum_exp=sum_exp,
        dtype=orig_dtype,
    )


def merge_two_partial_attentions(
    chunk_a: PartialAttentionChunk,
    chunk_b: PartialAttentionChunk,
) -> PartialAttentionChunk:
    """Merges two partial attention chunks using online softmax rescaling in FP32.

    Mathematically identical to computing softmax over chunk_a and chunk_b
    concatenated together. Zero accuracy loss.

    Args:
        chunk_a: First partial attention chunk.
        chunk_b: Second partial attention chunk.

    Returns:
        Merged PartialAttentionChunk representing the combined context.
    """
    target_device = chunk_b.accumulator.device
    if chunk_a.accumulator.device != target_device:
        chunk_a = chunk_a.to(target_device)

    max_a = chunk_a.max_score.to(torch.float32)
    max_b = chunk_b.max_score.to(torch.float32)

    # 1. Global maximum across both chunks
    m_global = torch.maximum(max_a, max_b)
    safe_m_global = torch.where(
        torch.isneginf(m_global), torch.zeros_like(m_global), m_global
    )

    # 2. Rescaling factors (guaranteed <= 1.0, zero risk of numerical overflow or -inf - -inf)
    alpha_a = torch.exp(max_a - safe_m_global)
    alpha_b = torch.exp(max_b - safe_m_global)

    # 3. Rescaled denominators in FP32 (mask out zero-weight partitions to prevent NaN * 0)
    sum_a = chunk_a.sum_exp.to(torch.float32)
    sum_b = chunk_b.sum_exp.to(torch.float32)
    sum_exp_global = torch.where(
        alpha_a == 0, torch.zeros_like(sum_a), sum_a * alpha_a
    ) + (torch.where(alpha_b == 0, torch.zeros_like(sum_b), sum_b * alpha_b))

    # 4. Rescaled numerators in FP32
    acc_a = chunk_a.accumulator.to(torch.float32)
    acc_b = chunk_b.accumulator.to(torch.float32)
    accumulator_global = torch.where(
        alpha_a == 0, torch.zeros_like(acc_a), acc_a * alpha_a
    ) + torch.where(alpha_b == 0, torch.zeros_like(acc_b), acc_b * alpha_b)

    target_dtype = chunk_b.dtype if chunk_b.dtype is not None else chunk_a.dtype

    return PartialAttentionChunk(
        accumulator=accumulator_global,
        max_score=m_global,
        sum_exp=sum_exp_global,
        dtype=target_dtype,
    )


def merge_partial_attentions(
    chunks: list[PartialAttentionChunk],
) -> PartialAttentionChunk:
    """Merges an arbitrary list of partial attention chunks sequentially."""
    if not chunks:
        raise ValueError("Cannot merge empty list of PartialAttentionChunks.")
    if len(chunks) == 1:
        return chunks[0]

    merged = chunks[0]
    for next_chunk in chunks[1:]:
        merged = merge_two_partial_attentions(merged, next_chunk)
    return merged


def finalize_attention(
    chunk: PartialAttentionChunk,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Normalizes the merged partial attention chunk in FP32 and casts to target dtype.

    Args:
        chunk: The merged (or single) PartialAttentionChunk.
        dtype: Optional explicit output dtype override. Defaults to `chunk.dtype`.

    Returns:
        Tensor of shape `[batch, num_heads, q_len, head_dim]`.
    """
    target_dtype = dtype or chunk.dtype or chunk.accumulator.dtype
    # Perform division in FP32 with epsilon guard, then downcast at the last moment
    denom = torch.clamp(chunk.sum_exp.to(torch.float32), min=1e-12)
    out = chunk.accumulator.to(torch.float32) / denom
    return out.to(target_dtype)
