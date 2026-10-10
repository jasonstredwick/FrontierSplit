"""HuggingFace Model Integration for FrontierSplit Disaggregated Inference.

Wraps any standard HuggingFace autoregressive causal language model (Llama,
Mistral, Qwen2, Gemma) to offload prompt KV caches to a remote Context Server
and generate tokens using Online Softmax attention merging.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from frontiersplit.online_softmax import (
    compute_partial_attention,
    finalize_attention,
    merge_two_partial_attentions,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from frontiersplit.context_server import ContextClient

logger = logging.getLogger("frontiersplit.hf_model")


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings (RoPE) to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # rotate half: [-x2, x1]
    rotate_half_q = torch.cat(
        (-q[..., q.shape[-1] // 2 :], q[..., : q.shape[-1] // 2]), dim=-1
    )
    rotate_half_k = torch.cat(
        (-k[..., k.shape[-1] // 2 :], k[..., : k.shape[-1] // 2]), dim=-1
    )

    q_embed = (q * cos) + (rotate_half_q * sin)
    k_embed = (k * cos) + (rotate_half_k * sin)
    return q_embed, k_embed


class DisaggregatedAttentionPatcher:
    """Replaces a HuggingFace attention forward pass with disaggregated merge logic."""

    def __init__(
        self,
        original_attn: nn.Module,
        layer_idx: int,
        context_client: ContextClient,
    ) -> None:
        self.original_attn = original_attn
        self.original_forward = original_attn.forward
        self.layer_idx = layer_idx
        self.context_client = context_client
        self.active_session_id: str | None = None

    def set_session(self, session_id: str | None) -> None:
        """Sets the active session ID for routing queries to the Context Server."""
        self.active_session_id = session_id

    def __call__(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        # If no remote session bound (e.g. during prefill), use native HF attention forward
        if self.active_session_id is None:
            return self.original_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        head_dim = self.original_attn.head_dim
        hidden_shape = (*input_shape, -1, head_dim)

        # 1. Project Q, K, V
        q = self.original_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.original_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.original_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # 2. Apply RoPE if position embeddings provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # 3. Update local volatile output cache with this token's K, V
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        # 4. Execute disaggregated attention merge over TCP
        scaling = getattr(self.original_attn, "scaling", 1.0 / (head_dim**0.5))

        prompt_chunk = self.context_client.query_partial_attention_sync(
            session_id=self.active_session_id,
            layer_idx=self.layer_idx,
            q=q,
            scale=scaling,
        )

        if past_key_values is None:
            attn_out = finalize_attention(prompt_chunk)
        else:
            # Compute local partial attention on volatile output tokens
            local_chunk = compute_partial_attention(q, k, v, scale=scaling)

            # Merge via Online Softmax
            merged = merge_two_partial_attentions(prompt_chunk, local_chunk)
            attn_out = finalize_attention(merged)

        # 5. Reshape and project out
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return self.original_attn.o_proj(attn_out), None


class DisaggregatedModel:
    """High-level wrapper that coordinates disaggregated inference for HuggingFace models."""

    def __init__(
        self,
        model: PreTrainedModel,
        context_client: ContextClient,
    ) -> None:
        self.model = model
        self.context_client = context_client
        self._patchers: list[DisaggregatedAttentionPatcher] = []
        self._patched = False
        self._patch_layers()

    def _patch_layers(self) -> None:
        """Installs disaggregated attention patchers across all model layers."""
        layers = getattr(self.model.model, "layers", [])
        for idx, layer in enumerate(layers):
            patcher = DisaggregatedAttentionPatcher(
                original_attn=layer.self_attn,
                layer_idx=idx,
                context_client=self.context_client,
            )
            layer.self_attn.forward = patcher
            self._patchers.append(patcher)
        self._patched = True

    def set_active_session(self, session_id: str | None) -> None:
        """Sets the active session ID for all patched attention layers."""
        for p in self._patchers:
            p.set_session(session_id)

    def prefill_and_offload(
        self,
        input_ids: torch.Tensor,
        session_id: str,
    ) -> tuple[torch.Tensor, int]:
        """Runs prefill on the prompt and offloads static prompt KV caches to Context Server.

        Args:
            input_ids: Prompt token tensor of shape `[1, prompt_len]`.
            session_id: Unique session identifier.

        Returns:
            Tuple of (first_generated_token_id, prompt_length).
        """
        # Disable disaggregation during prefill so standard full attention computes KV cache
        self.set_active_session(None)

        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
            prompt_cache = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        prompt_len = input_ids.shape[1]
        tokens = input_ids[0].tolist()

        # Offload each layer's static prompt KV cache to the remote Context Server
        for l_idx, layer in enumerate(prompt_cache.layers):
            k_prompt = layer.keys.clone()
            v_prompt = layer.values.clone()
            self.context_client.register_prompt_sync(
                session_id=session_id,
                layer_idx=l_idx,
                k=k_prompt,
                v=v_prompt,
                token_ids=tokens,
            )

        # Bind session to all patchers for subsequent decode steps
        self.set_active_session(session_id)
        return next_token, prompt_len

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        session_id: str = "default-session",
        use_prefix_cache: bool = True,
    ) -> list[int]:
        """Generates tokens synchronously using disaggregated attention merging.

        Args:
            input_ids: Prompt token tensor of shape `[1, prompt_len]`.
            max_new_tokens: Maximum number of new tokens to generate.
            session_id: Session identifier.
            use_prefix_cache: If True, checks for existing prefix cache on Context Server.

        Returns:
            List of generated token IDs (including the first generated token).
        """
        tokens = input_ids[0].tolist()
        prompt_len = len(tokens)

        # Check for warm prefix cache hit on Context Server
        cache_hit = False
        if use_prefix_cache:
            matched_len, _ = self.context_client.match_prefix_sync(
                tokens, session_id=session_id
            )
            if matched_len == prompt_len:
                cache_hit = True

        if cache_hit:
            # 100% prefix cache hit: skip full prefill compute
            self.set_active_session(session_id)
            last_tok = input_ids[:, -1:]
            pos_tensor = torch.tensor([[prompt_len - 1]], device=input_ids.device)
            with torch.no_grad():
                out_fast = self.model(
                    last_tok, position_ids=pos_tensor, use_cache=False
                )
                next_tok = out_fast.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        else:
            # 1. Prefill and offload prompt KV
            next_tok, prompt_len = self.prefill_and_offload(input_ids, session_id)

        generated_tokens = [next_tok.item()]
        cur_tok = next_tok

        # 2. Local volatile output cache (starts empty; never holds prompt tokens)
        local_cache = DynamicCache()

        # 3. Autoregressive decode loop
        try:
            for step in range(1, max_new_tokens):
                position = prompt_len + step - 1
                pos_tensor = torch.tensor([[position]], device=input_ids.device)

                with torch.no_grad():
                    out = self.model(
                        cur_tok,
                        past_key_values=local_cache,
                        use_cache=True,
                        position_ids=pos_tensor,
                    )
                    cur_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated_tokens.append(cur_tok.item())

        finally:
            # Clean up remote prompt KV cache
            self.context_client.release_session_sync(session_id)
            self.set_active_session(None)

        return generated_tokens

    async def prefill_and_offload_async(
        self,
        input_ids: torch.Tensor,
        session_id: str,
    ) -> tuple[torch.Tensor, int]:
        """Asynchronously runs prefill on the prompt and offloads KV caches."""
        self.set_active_session(None)

        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
            prompt_cache = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        prompt_len = input_ids.shape[1]
        tokens = input_ids[0].tolist()

        tasks = []
        for l_idx, layer in enumerate(prompt_cache.layers):
            k_prompt = layer.keys.clone()
            v_prompt = layer.values.clone()
            tasks.append(
                self.context_client.register_prompt(
                    session_id=session_id,
                    layer_idx=l_idx,
                    k=k_prompt,
                    v=v_prompt,
                    token_ids=tokens,
                )
            )
        await asyncio.gather(*tasks)

        self.set_active_session(session_id)
        return next_token, prompt_len

    async def generate_async(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        session_id: str = "default-session",
        use_prefix_cache: bool = True,
    ) -> list[int]:
        """Asynchronously generates tokens using disaggregated attention merging.

        Args:
            input_ids: Prompt token tensor of shape `[1, prompt_len]`.
            max_new_tokens: Maximum number of new tokens to generate.
            session_id: Session identifier.
            use_prefix_cache: If True, checks for existing prefix cache on Context Server.

        Returns:
            List of generated token IDs (including the first generated token).
        """
        tokens = input_ids[0].tolist()
        prompt_len = len(tokens)

        cache_hit = False
        if use_prefix_cache:
            matched_len, _ = await self.context_client.match_prefix(
                tokens, session_id=session_id
            )
            if matched_len == prompt_len:
                cache_hit = True

        if cache_hit:
            self.set_active_session(session_id)
            last_tok = input_ids[:, -1:]
            pos_tensor = torch.tensor([[prompt_len - 1]], device=input_ids.device)
            with torch.no_grad():
                out_fast = self.model(
                    last_tok, position_ids=pos_tensor, use_cache=False
                )
                next_tok = out_fast.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        else:
            next_tok, prompt_len = await self.prefill_and_offload_async(
                input_ids, session_id
            )

        generated_tokens = [next_tok.item()]
        cur_tok = next_tok
        local_cache = DynamicCache()

        try:
            for step in range(1, max_new_tokens):
                position = prompt_len + step - 1
                pos_tensor = torch.tensor([[position]], device=input_ids.device)

                with torch.no_grad():
                    out = self.model(
                        cur_tok,
                        past_key_values=local_cache,
                        use_cache=True,
                        position_ids=pos_tensor,
                    )
                    cur_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated_tokens.append(cur_tok.item())

        finally:
            await self.context_client.release_session(session_id)
            self.set_active_session(None)

        return generated_tokens


__all__ = [
    "DisaggregatedAttentionPatcher",
    "DisaggregatedModel",
]
