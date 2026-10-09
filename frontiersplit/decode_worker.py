"""Disaggregated Decode Worker for FrontierSplit.

Maintains a small, volatile local Key-Value (KV) cache for generated output tokens,
queries the remote Context Server for static prompt partial attention, and merges
them in real time via Online Softmax.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from frontiersplit.online_softmax import (
    compute_partial_attention,
    finalize_attention,
    merge_two_partial_attentions,
)

if TYPE_CHECKING:
    from frontiersplit.context_server import ContextClient

logger = logging.getLogger("frontiersplit.decode_worker")


class LocalOutputKVCache:
    """Manages the small, dynamic Key-Value buffer for output generation tokens.

    Unlike the prompt KV cache which is fixed-width and stored remotely, this
    buffer grows token-by-token during autoregressive decode, but only up to
    the maximum generated token limit (e.g. 512 to 2,048 tokens).
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        # Structure: {session_id: {layer_idx: (k_output_tensor, v_output_tensor)}}
        self._cache: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}

    def append(
        self,
        session_id: str,
        layer_idx: int,
        k_token: torch.Tensor,
        v_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Appends a newly generated token's key and value to the local cache.

        Args:
            session_id: Unique request identifier.
            layer_idx: The transformer layer index.
            k_token: Key tensor of shape `[batch, num_kv_heads, 1, head_dim]`.
            v_token: Value tensor of shape `[batch, num_kv_heads, 1, head_dim]`.

        Returns:
            Tuple of updated (k_all_output, v_all_output) tensors for this session and layer.
        """
        if session_id not in self._cache:
            self._cache[session_id] = {}

        layer_cache = self._cache[session_id].get(layer_idx)
        if layer_cache is None:
            k_all = k_token.detach().to(self.device)
            v_all = v_token.detach().to(self.device)
        else:
            k_prev, v_prev = layer_cache
            k_all = torch.cat([k_prev, k_token.detach().to(self.device)], dim=-2)
            v_all = torch.cat([v_prev, v_token.detach().to(self.device)], dim=-2)

        self._cache[session_id][layer_idx] = (k_all, v_all)
        return k_all, v_all

    def get(
        self,
        session_id: str,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Returns the local output KV cache for a session and layer, or None."""
        session = self._cache.get(session_id)
        if session is None:
            return None
        return session.get(layer_idx)

    def release_session(self, session_id: str) -> bool:
        """Evicts the local output cache for a completed session."""
        if session_id in self._cache:
            del self._cache[session_id]
            return True
        return False

    def clear(self) -> None:
        """Clears all session caches."""
        self._cache.clear()


class DisaggregatedAttention(nn.Module):
    """Attention layer module that integrates with a remote Context Server.

    Computes partial attention against local output tokens while concurrently
    fetching partial attention against the remote prompt KV cache, stitching
    both via Online Softmax.
    """

    def __init__(
        self,
        layer_idx: int,
        context_client: ContextClient,
        kv_cache: LocalOutputKVCache | None = None,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.context_client = context_client
        self.kv_cache = kv_cache or LocalOutputKVCache()
        self.scale = scale

    async def forward_decode_step(
        self,
        session_id: str,
        q: torch.Tensor,
        k_token: torch.Tensor,
        v_token: torch.Tensor,
    ) -> torch.Tensor:
        """Performs a single decode step with disaggregated context merging.

        Args:
            session_id: The active session identifier.
            q: Query tensor for the new token: `[batch, num_heads, 1, head_dim]`.
            k_token: Key tensor for the new token: `[batch, num_kv_heads, 1, head_dim]`.
            v_token: Value tensor for the new token: `[batch, num_kv_heads, 1, head_dim]`.

        Returns:
            Attention output tensor of shape `[batch, num_heads, 1, head_dim]`.
        """
        # 1. Asynchronously dispatch query to the remote Context Server (8 KB payload)
        remote_prompt_task = asyncio.create_task(
            self.context_client.query_partial_attention(
                session_id=session_id,
                layer_idx=self.layer_idx,
                q=q,
                scale=self.scale,
            )
        )

        # 2. Concurrently append token to local output cache and compute local partial attention
        k_local, v_local = self.kv_cache.append(
            session_id=session_id,
            layer_idx=self.layer_idx,
            k_token=k_token,
            v_token=v_token,
        )
        local_output_chunk = compute_partial_attention(
            q=q,
            k=k_local,
            v=v_local,
            scale=self.scale,
        )

        # 3. Await the remote prompt partial attention chunk
        remote_prompt_chunk = await remote_prompt_task

        # 4. Merge prompt and output chunks via Online Softmax (zero accuracy loss)
        merged_chunk = merge_two_partial_attentions(
            chunk_a=remote_prompt_chunk,
            chunk_b=local_output_chunk,
        )

        # 5. Finalize normalized attention
        return finalize_attention(merged_chunk)


class DecodeWorker:
    """Orchestrates disaggregated attention layers for a decode machine."""

    def __init__(
        self,
        context_client: ContextClient,
        num_layers: int = 1,
        device: str = "cpu",
    ) -> None:
        self.context_client = context_client
        self.device = torch.device(device)
        self.num_layers = num_layers
        self.kv_cache = LocalOutputKVCache(device=device)

        self.layers: list[DisaggregatedAttention] = [
            DisaggregatedAttention(
                layer_idx=i,
                context_client=context_client,
                kv_cache=self.kv_cache,
            )
            for i in range(num_layers)
        ]

    async def forward_layer(
        self,
        session_id: str,
        layer_idx: int,
        q: torch.Tensor,
        k_token: torch.Tensor,
        v_token: torch.Tensor,
    ) -> torch.Tensor:
        """Executes a single layer's disaggregated attention."""
        layer = self.layers[layer_idx]
        return await layer.forward_decode_step(session_id, q, k_token, v_token)

    async def end_session(self, session_id: str) -> None:
        """Releases session KV caches both locally and on the remote Context Server."""
        self.kv_cache.release_session(session_id)
        await self.context_client.release_session(session_id)


__all__ = [
    "DecodeWorker",
    "DisaggregatedAttention",
    "LocalOutputKVCache",
]
