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

    Attributes:
        accumulator: Unnormalized weighted value accumulator of shape
            `[batch, num_heads, q_len, head_dim]`.
        max_score: Maximum attention logit per query position of shape
            `[batch, num_heads, q_len, 1]`.
        sum_exp: Sum of exponentials (denominator) of shape
            `[batch, num_heads, q_len, 1]`.
    """

    accumulator: torch.Tensor
    max_score: torch.Tensor
    sum_exp: torch.Tensor

    def to(self, device: torch.device | str) -> PartialAttentionChunk:
        """Moves chunk tensors to the specified device."""
        return PartialAttentionChunk(
            accumulator=self.accumulator.to(device),
            max_score=self.max_score.to(device),
            sum_exp=self.sum_exp.to(device),
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

    # Compute raw scaled dot-product scores: [batch, num_heads, q_len, kv_len]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if attention_mask is not None:
        scores = scores + attention_mask

    # Numerically stable online softmax tracking
    # 1. Local maximum per query: [batch, num_heads, q_len, 1]
    max_score = torch.max(scores, dim=-1, keepdim=True).values

    # 2. Unnormalized exponentiated probabilities: [batch, num_heads, q_len, kv_len]
    p = torch.exp(scores - max_score)

    # 3. Sum of exponentials (denominator): [batch, num_heads, q_len, 1]
    sum_exp = torch.sum(p, dim=-1, keepdim=True)

    # 4. Partial accumulator (numerator): [batch, num_heads, q_len, head_dim]
    accumulator = torch.matmul(p, v)

    return PartialAttentionChunk(
        accumulator=accumulator,
        max_score=max_score,
        sum_exp=sum_exp,
    )


def merge_two_partial_attentions(
    chunk_a: PartialAttentionChunk,
    chunk_b: PartialAttentionChunk,
) -> PartialAttentionChunk:
    """Merges two partial attention chunks using online softmax rescaling.

    Mathematically identical to computing softmax over chunk_a and chunk_b
    concatenated together. Zero accuracy loss.

    Args:
        chunk_a: First partial attention chunk.
        chunk_b: Second partial attention chunk.

    Returns:
        Merged PartialAttentionChunk representing the combined context.
    """
    # 1. Global maximum across both chunks
    m_global = torch.maximum(chunk_a.max_score, chunk_b.max_score)

    # 2. Rescaling factors (guaranteed <= 1.0, zero risk of numerical overflow)
    alpha_a = torch.exp(chunk_a.max_score - m_global)
    alpha_b = torch.exp(chunk_b.max_score - m_global)

    # 3. Rescaled denominators
    sum_exp_global = (chunk_a.sum_exp * alpha_a) + (chunk_b.sum_exp * alpha_b)

    # 4. Rescaled numerators
    accumulator_global = (chunk_a.accumulator * alpha_a) + (
        chunk_b.accumulator * alpha_b
    )

    return PartialAttentionChunk(
        accumulator=accumulator_global,
        max_score=m_global,
        sum_exp=sum_exp_global,
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


def finalize_attention(chunk: PartialAttentionChunk) -> torch.Tensor:
    """Normalizes the merged partial attention chunk to yield the final attention output.

    Args:
        chunk: The merged (or single) PartialAttentionChunk.

    Returns:
        Tensor of shape `[batch, num_heads, q_len, head_dim]`.
    """
    # Add small epsilon to denominator to prevent division by zero in masked-out padding
    denom = torch.clamp(chunk.sum_exp, min=1e-12)
    return chunk.accumulator / denom
