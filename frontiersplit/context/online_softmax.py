"""Online Softmax Attention primitives for disaggregated inference.

This module implements exact, numerically stable online softmax chunking for
attention mechanisms (Dao et al. 2022, Milakov & Gimelshein 2018).

It allows attention over a long context to be computed across separate chunks
(e.g., a static prompt cache on a remote Context Server and a dynamic local
output cache on a Decode Worker) and merged into a mathematically exact final
attention output with zero accuracy loss.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
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


@contextlib.contextmanager
def _fp32_precision_guard(device: torch.device) -> Iterator[None]:
    """Disables PyTorch autocast so explicit FP32 attention math is not downcast."""
    with torch.autocast(device_type=device.type, enabled=False):
        if device.type != "cpu" and torch.is_autocast_cpu_enabled():
            with torch.autocast(device_type="cpu", enabled=False):
                yield
        else:
            yield


def _compute_partial_attention_block(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attention_mask: torch.Tensor | None,
    orig_dtype: torch.dtype,
) -> PartialAttentionChunk:
    """Computes partial attention over a single bounded workspace block."""
    batch_size, num_heads, q_len, head_dim = q.shape
    _, num_kv_heads, kv_len, _ = k.shape

    # Expand KV heads for GQA locally on this block (bounded to block size, e.g. <=1024 tokens)
    if num_heads != num_kv_heads:
        n_rep = num_heads // num_kv_heads
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

    with _fp32_precision_guard(q.device):
        scores = (
            torch.matmul(q.to(torch.float32), k.transpose(-2, -1).to(torch.float32))
            * scale
        )
        if attention_mask is not None:
            scores = scores + attention_mask.to(torch.float32)

        max_score = torch.max(scores, dim=-1, keepdim=True).values
        safe_max = torch.where(
            torch.isneginf(max_score), torch.zeros_like(max_score), max_score
        )
        p = torch.exp(scores - safe_max)
        sum_exp = torch.sum(p, dim=-1, keepdim=True)
        accumulator = torch.matmul(p, v.to(torch.float32))
        accumulator = torch.where(
            sum_exp == 0, torch.zeros_like(accumulator), accumulator
        )

    return PartialAttentionChunk(
        accumulator=accumulator,
        max_score=max_score,
        sum_exp=sum_exp,
        dtype=orig_dtype,
    )


def compute_partial_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    attention_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    sliding_window: int | None = None,
    q_pos_offset: int | None = None,
    k_pos_offset: int | None = None,
    query_chunk_size: int = 128,
    kv_chunk_size: int = 1024,
) -> PartialAttentionChunk:
    """Computes partial attention against a chunk of keys and values.

    Tiles query and KV dimensions to ensure intermediate workspace matrices
    (attention scores, GQA expansion, and causal masks) remain bounded in memory.

    Args:
        q: Query tensor of shape `[batch, num_heads, q_len, head_dim]`.
        k: Key tensor of shape `[batch, num_kv_heads, kv_len, head_dim]`.
        v: Value tensor of shape `[batch, num_kv_heads, kv_len, head_dim]`.
        scale: Softmax temperature scale factor (default: 1 / sqrt(head_dim)).
        attention_mask: Optional mask broadcastable to `[batch, 1, q_len, kv_len]`.
        is_causal: If True, applies causal lower-triangular masking within tiles.
        sliding_window: Optional Mistral-style local attention window size.
        q_pos_offset: Optional starting position offset for queries (default: kv_len - q_len).
        k_pos_offset: Optional starting position offset for keys (default: 0).
        query_chunk_size: Maximum query tokens per tiled workspace block (default: 128).
        kv_chunk_size: Maximum KV tokens per tiled workspace block (default: 1024).

    Returns:
        A PartialAttentionChunk containing (accumulator, max_score, sum_exp).
    """
    batch_size, num_heads, q_len, head_dim = q.shape
    _, num_kv_heads, kv_len, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    orig_dtype = q.dtype
    q_base = (kv_len - q_len) if q_pos_offset is None else q_pos_offset
    k_base = 0 if k_pos_offset is None else k_pos_offset

    if q_len == 0 or kv_len == 0:
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

    # For bounded workloads (e.g. single decode query q_len=1 against <=1024 cache),
    # evaluate directly without tile iteration overhead.
    if q_len <= query_chunk_size and kv_len <= kv_chunk_size:
        local_mask = attention_mask
        if is_causal or sliding_window is not None:
            q_pos = q_base + torch.arange(
                q_len, device=q.device
            ).unsqueeze(1)
            k_pos = k_base + torch.arange(kv_len, device=q.device).unsqueeze(0)
            masked_out = None
            if is_causal:
                masked_out = k_pos > q_pos
            if sliding_window is not None:
                sw_masked = k_pos <= (q_pos - sliding_window)
                masked_out = (
                    sw_masked if masked_out is None else (masked_out | sw_masked)
                )
            if masked_out is not None and masked_out.any():
                tile_mask = torch.where(
                    masked_out,
                    torch.tensor(float("-inf"), dtype=torch.float32, device=q.device),
                    torch.tensor(0.0, dtype=torch.float32, device=q.device),
                ).view(1, 1, q_len, kv_len)
                local_mask = (
                    tile_mask
                    if local_mask is None
                    else local_mask.to(torch.float32) + tile_mask
                )
        return _compute_partial_attention_block(
            q, k, v, scale, local_mask, orig_dtype
        )

    # Tile both query and KV dimensions to strictly bound intermediate memory workspace:
    # workspace tensor size is at most [batch, num_heads, query_chunk_size, kv_chunk_size].
    q_chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for q_start in range(0, q_len, query_chunk_size):
        q_end = min(q_start + query_chunk_size, q_len)
        q_block_len = q_end - q_start
        q_block = q[:, :, q_start:q_end, :]

        acc_q: torch.Tensor | None = None
        max_q: torch.Tensor | None = None
        sum_q: torch.Tensor | None = None

        last_q_pos = q_base + q_end - 1

        for kv_start in range(0, kv_len, kv_chunk_size):
            kv_end = min(kv_start + kv_chunk_size, kv_len)
            kv_block_len = kv_end - kv_start
            k_block = k[:, :, kv_start:kv_end, :]
            v_block = v[:, :, kv_start:kv_end, :]

            # Independent axis mask slicing preserving singleton dimensions
            mask_block: torch.Tensor | None = None
            if attention_mask is not None:
                if attention_mask.dim() >= 2:
                    mask_q = attention_mask.shape[-2]
                    mask_k = attention_mask.shape[-1]
                    q_sl = slice(q_start, q_end) if mask_q == q_len else slice(None)
                    k_sl = slice(kv_start, kv_end) if mask_k == kv_len else slice(None)
                    mask_block = attention_mask[..., q_sl, k_sl]
                elif attention_mask.dim() == 1:
                    k_sl = (
                        slice(kv_start, kv_end)
                        if attention_mask.shape[-1] == kv_len
                        else slice(None)
                    )
                    mask_block = attention_mask[k_sl]
                else:
                    mask_block = attention_mask

            # Fast check: if strictly upper triangular (all keys in tile are in the future of all queries in tile)
            first_k_pos = k_base + kv_start
            if is_causal and first_k_pos > last_q_pos:
                block_res = PartialAttentionChunk(
                    accumulator=torch.zeros(
                        (batch_size, num_heads, q_block_len, head_dim),
                        dtype=torch.float32,
                        device=q.device,
                    ),
                    max_score=torch.full(
                        (batch_size, num_heads, q_block_len, 1),
                        float("-inf"),
                        dtype=torch.float32,
                        device=q.device,
                    ),
                    sum_exp=torch.zeros(
                        (batch_size, num_heads, q_block_len, 1),
                        dtype=torch.float32,
                        device=q.device,
                    ),
                    dtype=orig_dtype,
                )
            else:
                if is_causal or sliding_window is not None:
                    q_pos = (q_base + q_start) + torch.arange(
                        q_block_len, device=q.device
                    ).unsqueeze(1)
                    k_pos = (k_base + kv_start) + torch.arange(
                        kv_block_len, device=q.device
                    ).unsqueeze(0)

                    tile_masked_out = None
                    if is_causal:
                        tile_masked_out = k_pos > q_pos
                    if sliding_window is not None:
                        sw_masked = k_pos <= (q_pos - sliding_window)
                        tile_masked_out = (
                            sw_masked
                            if tile_masked_out is None
                            else (tile_masked_out | sw_masked)
                        )

                    if tile_masked_out is not None and tile_masked_out.any():
                        if tile_masked_out.all():
                            block_res = PartialAttentionChunk(
                                accumulator=torch.zeros(
                                    (batch_size, num_heads, q_block_len, head_dim),
                                    dtype=torch.float32,
                                    device=q.device,
                                ),
                                max_score=torch.full(
                                    (batch_size, num_heads, q_block_len, 1),
                                    float("-inf"),
                                    dtype=torch.float32,
                                    device=q.device,
                                ),
                                sum_exp=torch.zeros(
                                    (batch_size, num_heads, q_block_len, 1),
                                    dtype=torch.float32,
                                    device=q.device,
                                ),
                                dtype=orig_dtype,
                            )
                        else:
                            tile_mask = torch.where(
                                tile_masked_out,
                                torch.tensor(
                                    float("-inf"),
                                    dtype=torch.float32,
                                    device=q.device,
                                ),
                                torch.tensor(
                                    0.0,
                                    dtype=torch.float32,
                                    device=q.device,
                                ),
                            ).view(1, 1, q_block_len, kv_block_len)
                            mask_block = (
                                tile_mask
                                if mask_block is None
                                else mask_block.to(torch.float32) + tile_mask
                            )
                            block_res = _compute_partial_attention_block(
                                q_block,
                                k_block,
                                v_block,
                                scale,
                                mask_block,
                                orig_dtype,
                            )
                    else:
                        block_res = _compute_partial_attention_block(
                            q_block,
                            k_block,
                            v_block,
                            scale,
                            mask_block,
                            orig_dtype,
                        )
                else:
                    block_res = _compute_partial_attention_block(
                        q_block,
                        k_block,
                        v_block,
                        scale,
                        mask_block,
                        orig_dtype,
                    )

            if acc_q is None or max_q is None or sum_q is None:
                acc_q = block_res.accumulator
                max_q = block_res.max_score
                sum_q = block_res.sum_exp
            else:
                with _fp32_precision_guard(q.device):
                    new_max = torch.maximum(max_q, block_res.max_score)
                    safe_new_max = torch.where(
                        torch.isneginf(new_max), torch.zeros_like(new_max), new_max
                    )
                    alpha = torch.exp(
                        torch.where(
                            torch.isneginf(max_q),
                            torch.full_like(max_q, -1e9),
                            max_q - safe_new_max,
                        )
                    )
                    beta = torch.exp(
                        torch.where(
                            torch.isneginf(block_res.max_score),
                            torch.full_like(block_res.max_score, -1e9),
                            block_res.max_score - safe_new_max,
                        )
                    )
                    acc_q = acc_q * alpha + block_res.accumulator * beta
                    sum_q = sum_q * alpha + block_res.sum_exp * beta
                    max_q = new_max

        assert acc_q is not None and max_q is not None and sum_q is not None
        q_chunks.append((acc_q, max_q, sum_q))

    # Concatenate across query blocks along sequence dimension (dim=2)
    accumulator = torch.cat([c[0] for c in q_chunks], dim=2)
    max_score = torch.cat([c[1] for c in q_chunks], dim=2)
    sum_exp = torch.cat([c[2] for c in q_chunks], dim=2)

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

    with _fp32_precision_guard(target_device):
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
    with _fp32_precision_guard(chunk.accumulator.device):
        # Perform division in FP32 with epsilon guard, then downcast at the last moment
        denom = torch.clamp(chunk.sum_exp.to(torch.float32), min=1e-12)
        out = chunk.accumulator.to(torch.float32) / denom
    return out.to(target_dtype)
