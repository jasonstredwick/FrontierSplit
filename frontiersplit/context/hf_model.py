"""HuggingFace Model Integration for FrontierSplit Disaggregated Inference.

Wraps any standard HuggingFace autoregressive causal language model (Llama,
Mistral, Qwen2, Gemma) to offload prompt KV caches to a remote Context Server
and generate tokens using Online Softmax attention merging.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import threading
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from frontiersplit.context.online_softmax import (
    compute_partial_attention,
    finalize_attention,
    merge_two_partial_attentions,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from frontiersplit.context.context_server import ContextClient

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

    q_embed = ((q * cos) + (rotate_half_q * sin)).to(q.dtype)
    k_embed = ((k * cos) + (rotate_half_k * sin)).to(k.dtype)
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
        self.active_context_len: int | None = None
        self._session_var: contextvars.ContextVar[
            tuple[str | None, int | None] | None
        ] = contextvars.ContextVar(
            f"frontiersplit_attn_session_{id(self)}", default=None
        )

    def set_session(
        self, session_id: str | None, context_len: int | None = None
    ) -> None:
        """Sets the active session ID and optional prefix slice length."""
        self.active_session_id = session_id
        self.active_context_len = context_len
        self._session_var.set((session_id, context_len))

    def _get_active_session(self) -> tuple[str | None, int | None]:
        """Returns the current task's (session_id, context_len) with instance fallback."""
        ctx_val = self._session_var.get()
        if ctx_val is not None:
            return ctx_val
        return self.active_session_id, self.active_context_len

    def _get_sliding_window(self) -> int | None:
        """Returns the sliding window size for this attention layer, if active."""
        cfg = getattr(self.original_attn, "config", None)
        if cfg is not None:
            layer_types = getattr(cfg, "layer_types", None)
            if (
                isinstance(layer_types, (list, tuple))
                and self.layer_idx < len(layer_types)
                and "sliding" not in str(layer_types[self.layer_idx])
            ):
                return None
        sw = getattr(self.original_attn, "sliding_window", None)
        if sw is None and cfg is not None:
            sw = getattr(cfg, "sliding_window", None)
        return int(sw) if isinstance(sw, int) and sw > 0 else None

    def __call__(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        session_id, context_len = self._get_active_session()

        # If no remote session bound (e.g. during prefill), use native HF attention forward
        if session_id is None:
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
        sliding_window = self._get_sliding_window()

        if past_key_values is None:
            prompt_chunk = self.context_client.query_partial_attention_sync(
                session_id=session_id,
                layer_idx=self.layer_idx,
                q=q,
                scale=scaling,
                context_len=context_len,
                sliding_window=sliding_window,
                past_local_len=0,
            ).to(q.device)
            attn_out = finalize_attention(prompt_chunk)
        else:
            q_len = q.shape[-2]
            kv_len = k.shape[-2]
            local_mask = attention_mask
            if q_len > 1 or (sliding_window is not None and kv_len > sliding_window):
                q_idx = torch.arange(kv_len - q_len, kv_len, device=q.device).unsqueeze(
                    1
                )
                k_idx = torch.arange(0, kv_len, device=q.device).unsqueeze(0)
                masked_out = k_idx > q_idx
                if sliding_window is not None:
                    masked_out = masked_out | (k_idx <= (q_idx - sliding_window))
                if masked_out.any():
                    built_mask = torch.where(
                        masked_out,
                        torch.full(
                            (q_len, kv_len),
                            float("-inf"),
                            device=q.device,
                            dtype=torch.float32,
                        ),
                        torch.zeros(
                            (q_len, kv_len),
                            device=q.device,
                            dtype=torch.float32,
                        ),
                    ).view(1, 1, q_len, kv_len)
                    local_mask = (
                        built_mask
                        if local_mask is None
                        else local_mask.to(torch.float32) + built_mask
                    )

            # Overlap remote TCP query with local partial attention compute
            sync_lock = getattr(
                self.context_client, "_sync_lock", contextlib.nullcontext()
            )
            if hasattr(self.context_client, "send_query_partial_sync") and hasattr(
                self.context_client, "recv_query_partial_sync"
            ):
                with sync_lock:
                    try:
                        self.context_client.send_query_partial_sync(
                            session_id=session_id,
                            layer_idx=self.layer_idx,
                            q=q,
                            scale=scaling,
                            context_len=context_len,
                            sliding_window=sliding_window,
                            past_local_len=kv_len,
                        )
                        # Compute local partial attention while Context Server computes remote chunk
                        local_chunk = compute_partial_attention(
                            q, k, v, scale=scaling, attention_mask=local_mask
                        )
                        prompt_chunk = self.context_client.recv_query_partial_sync(
                            device=q.device
                        )
                    except BaseException:
                        if hasattr(self.context_client, "close_sync"):
                            self.context_client.close_sync()
                        raise
            else:
                prompt_chunk = self.context_client.query_partial_attention_sync(
                    session_id=session_id,
                    layer_idx=self.layer_idx,
                    q=q,
                    scale=scaling,
                    context_len=context_len,
                    sliding_window=sliding_window,
                    past_local_len=kv_len,
                ).to(q.device)
                local_chunk = compute_partial_attention(
                    q, k, v, scale=scaling, attention_mask=local_mask
                )

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
        self._forward_lock = threading.RLock()
        self.last_finish_reason: str = "stop"
        self._finish_reason_var: contextvars.ContextVar[str] = contextvars.ContextVar(
            f"frontiersplit_finish_reason_{id(self)}", default="stop"
        )
        self._patched = False
        self._patch_layers()

    def _set_finish_reason(self, reason: str) -> None:
        """Records the finish reason ('stop' or 'length') for the current generation."""
        self.last_finish_reason = reason
        self._finish_reason_var.set(reason)

    def get_finish_reason(self) -> str:
        """Returns the finish reason ('stop' or 'length') of the most recent generation."""
        return self._finish_reason_var.get()

    def _get_eos_token_ids(self) -> set[int]:
        """Extracts configured EOS token IDs from model config and generation_config."""
        eos_ids: set[int] = set()
        for cfg in (
            getattr(self.model, "config", None),
            getattr(self.model, "generation_config", None),
        ):
            if cfg is None:
                continue
            val = getattr(cfg, "eos_token_id", None)
            if isinstance(val, int):
                eos_ids.add(val)
            elif isinstance(val, (list, tuple, set)):
                eos_ids.update(int(x) for x in val if x is not None)
        return eos_ids

    @staticmethod
    def _validate_sampling_params(temperature: float, top_p: float) -> None:
        """Validates temperature and top_p sampling parameters."""
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0, got {temperature}")
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0.0, 1.0], got {top_p}")

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Selects the next token ID from final-position logits using greedy or nucleus sampling."""
        self._validate_sampling_params(temperature, top_p)
        next_logits = logits[:, -1, :].float()
        if temperature == 0.0:
            return next_logits.argmax(dim=-1, keepdim=True)

        scaled_logits = next_logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                scaled_logits, descending=True, dim=-1
            )
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            scaled_logits = scaled_logits.masked_fill(indices_to_remove, float("-inf"))

        probs = torch.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

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

    def set_active_session(
        self, session_id: str | None, context_len: int | None = None
    ) -> None:
        """Sets the active session ID and optional prefix slice for all patched layers."""
        for p in self._patchers:
            p.set_session(session_id, context_len=context_len)

    def prefill_and_offload(
        self,
        input_ids: torch.Tensor,
        session_id: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> tuple[torch.Tensor, int]:
        """Runs prefill on the prompt and offloads static prompt KV caches to Context Server.

        Args:
            input_ids: Prompt token tensor of shape `[1, prompt_len]`.
            session_id: Unique session identifier.
            temperature: Sampling temperature (0.0 for greedy decoding).
            top_p: Nucleus sampling probability threshold in (0.0, 1.0].

        Returns:
            Tuple of (first_generated_token_id, prompt_length).
        """
        self._validate_sampling_params(temperature, top_p)
        prompt_cache = DynamicCache()
        with self._forward_lock, torch.no_grad():
            self.set_active_session(None)
            outputs = self.model(
                input_ids, past_key_values=prompt_cache, use_cache=True
            )
            prompt_cache = outputs.past_key_values
            next_token = self._sample_next_token(
                outputs.logits, temperature=temperature, top_p=top_p
            )

        prompt_len = input_ids.shape[1]
        tokens = input_ids[0].tolist()
        num_layers = len(prompt_cache.layers)
        upload_id = uuid.uuid4().hex

        # Offload each layer's static prompt KV cache to the remote Context Server
        for l_idx, layer in enumerate(prompt_cache.layers):
            k_prompt = layer.keys.clone()
            v_prompt = layer.values.clone()
            self.context_client.register_prompt_sync(
                session_id=session_id,
                layer_idx=l_idx,
                k=k_prompt,
                v=v_prompt,
                token_ids=tokens if l_idx == num_layers - 1 else None,
                num_layers=num_layers,
                upload_id=upload_id,
            )

        # Bind session to all patchers for subsequent decode steps
        self.set_active_session(session_id)
        return next_token, prompt_len

    def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        session_id: str = "default-session",
        use_prefix_cache: bool = True,
        release_on_completion: bool = True,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Iterator[int]:
        """Synchronously streams generated tokens one by one as they are produced.

        Args:
            input_ids: Prompt token tensor of shape `[1, prompt_len]`.
            max_new_tokens: Maximum number of new tokens to generate.
            session_id: Session identifier.
            use_prefix_cache: If True, checks for existing prefix cache on Context Server.
            release_on_completion: If True, evicts session KV cache on completion.
                Set to False for multi-turn sessions with sliding TTL.
            temperature: Sampling temperature (0.0 for greedy decoding).
            top_p: Nucleus sampling probability threshold in (0.0, 1.0].

        Yields:
            Generated token IDs one by one (including the first generated token).
        """
        self._validate_sampling_params(temperature, top_p)
        if max_new_tokens <= 0:
            self._set_finish_reason("length")
            return

        tokens = input_ids[0].tolist()
        prompt_len = len(tokens)
        eos_ids = self._get_eos_token_ids()
        self._set_finish_reason("length")

        try:
            # Check for warm prefix cache hit on Context Server
            cache_hit = False
            matched_len = 0
            if use_prefix_cache:
                matched_len, _ = self.context_client.match_prefix_sync(
                    tokens, session_id=session_id
                )
                if matched_len == prompt_len:
                    cache_hit = True

            if cache_hit:
                # 100% prefix cache hit: skip full prefill compute
                last_tok = input_ids[:, -1:]
                pos_tensor = torch.tensor([[prompt_len - 1]], device=input_ids.device)
                with self._forward_lock, torch.no_grad():
                    self.set_active_session(session_id)
                    out_fast = self.model(
                        last_tok, position_ids=pos_tensor, use_cache=False
                    )
                    next_tok = self._sample_next_token(
                        out_fast.logits, temperature=temperature, top_p=top_p
                    )
            elif 0 < matched_len < prompt_len:
                # Disaggregated Delta Prefill:
                # Tokens [0..matched_len] are already warm on Context Server.
                # Suffix tokens [matched_len..prompt_len] run forward pass with disaggregated attention.
                suffix_ids = input_ids[:, matched_len:]
                suffix_pos = torch.arange(
                    matched_len, prompt_len, device=input_ids.device
                ).unsqueeze(0)
                suffix_cache = DynamicCache()
                with self._forward_lock, torch.no_grad():
                    self.set_active_session(session_id, context_len=matched_len)
                    out_suffix = self.model(
                        suffix_ids,
                        past_key_values=suffix_cache,
                        position_ids=suffix_pos,
                        use_cache=True,
                    )
                    next_tok = self._sample_next_token(
                        out_suffix.logits, temperature=temperature, top_p=top_p
                    )

                # Append suffix KV activations to the Context Server
                suffix_tokens = tokens[matched_len:]
                num_layers = len(suffix_cache.layers)
                upload_id = uuid.uuid4().hex
                for l_idx, layer in enumerate(suffix_cache.layers):
                    self.context_client.append_kv_sync(
                        session_id=session_id,
                        layer_idx=l_idx,
                        k_delta=layer.keys,
                        v_delta=layer.values,
                        token_ids=suffix_tokens if l_idx == num_layers - 1 else None,
                        num_layers=num_layers,
                        upload_id=upload_id,
                    )

                # Re-bind session with full context length for subsequent decode steps
                self.set_active_session(session_id, context_len=None)
            else:
                # 1. Prefill and offload prompt KV
                next_tok, prompt_len = self.prefill_and_offload(
                    input_ids,
                    session_id,
                    temperature=temperature,
                    top_p=top_p,
                )

            first_tok_id = int(next_tok.item())
            if first_tok_id in eos_ids:
                self._set_finish_reason("stop")
                yield first_tok_id
                return

            yield first_tok_id
            cur_tok = next_tok

            # 2. Local volatile output cache (starts empty; never holds prompt tokens)
            local_cache = DynamicCache()

            # 3. Autoregressive decode loop
            for step in range(1, max_new_tokens):
                position = prompt_len + step - 1
                pos_tensor = torch.tensor([[position]], device=input_ids.device)

                with self._forward_lock, torch.no_grad():
                    self.set_active_session(session_id, context_len=None)
                    out = self.model(
                        cur_tok,
                        past_key_values=local_cache,
                        use_cache=True,
                        position_ids=pos_tensor,
                    )
                    cur_tok = self._sample_next_token(
                        out.logits, temperature=temperature, top_p=top_p
                    )
                tok_id = int(cur_tok.item())
                if tok_id in eos_ids:
                    self._set_finish_reason("stop")
                    yield tok_id
                    return
                yield tok_id

        finally:
            if release_on_completion:
                with contextlib.suppress(Exception):
                    self.context_client.release_session_sync(session_id)
            self.set_active_session(None)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        session_id: str = "default-session",
        use_prefix_cache: bool = True,
        release_on_completion: bool = True,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[int]:
        """Generates tokens synchronously using disaggregated attention merging.

        Args:
            input_ids: Prompt token tensor of shape `[1, prompt_len]`.
            max_new_tokens: Maximum number of new tokens to generate.
            session_id: Session identifier.
            use_prefix_cache: If True, checks for existing prefix cache on Context Server.
            release_on_completion: If True, releases session KV cache on completion.
            temperature: Sampling temperature (0.0 for greedy decoding).
            top_p: Nucleus sampling probability threshold in (0.0, 1.0].

        Returns:
            List of generated token IDs (including the first generated token).
        """
        return list(
            self.generate_stream(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                session_id=session_id,
                use_prefix_cache=use_prefix_cache,
                release_on_completion=release_on_completion,
                temperature=temperature,
                top_p=top_p,
            )
        )

    async def prefill_and_offload_async(
        self,
        input_ids: torch.Tensor,
        session_id: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> tuple[torch.Tensor, int]:
        """Asynchronously runs prefill on the prompt and offloads KV caches."""
        self._validate_sampling_params(temperature, top_p)
        prompt_cache = DynamicCache()
        with self._forward_lock, torch.no_grad():
            self.set_active_session(None)
            outputs = self.model(
                input_ids, past_key_values=prompt_cache, use_cache=True
            )
            prompt_cache = outputs.past_key_values
            next_token = self._sample_next_token(
                outputs.logits, temperature=temperature, top_p=top_p
            )

        prompt_len = input_ids.shape[1]
        tokens = input_ids[0].tolist()
        num_layers = len(prompt_cache.layers)
        upload_id = uuid.uuid4().hex

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
                    token_ids=tokens if l_idx == num_layers - 1 else None,
                    num_layers=num_layers,
                    upload_id=upload_id,
                )
            )
        await asyncio.gather(*tasks)

        self.set_active_session(session_id)
        return next_token, prompt_len

    async def generate_stream_async(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        session_id: str = "default-session",
        use_prefix_cache: bool = True,
        release_on_completion: bool = True,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> AsyncIterator[int]:
        """Asynchronously streams generated tokens one by one as they are produced.

        Args:
            input_ids: Prompt token tensor of shape `[1, prompt_len]`.
            max_new_tokens: Maximum number of new tokens to generate.
            session_id: Session identifier.
            use_prefix_cache: If True, checks for existing prefix cache on Context Server.
            release_on_completion: If True, evicts session KV cache on completion.
            temperature: Sampling temperature (0.0 for greedy decoding).
            top_p: Nucleus sampling probability threshold in (0.0, 1.0].

        Yields:
            Generated token IDs one by one (including the first generated token).
        """
        self._validate_sampling_params(temperature, top_p)
        if max_new_tokens <= 0:
            self._set_finish_reason("length")
            return

        tokens = input_ids[0].tolist()
        prompt_len = len(tokens)
        eos_ids = self._get_eos_token_ids()
        self._set_finish_reason("length")

        try:
            cache_hit = False
            matched_len = 0
            if use_prefix_cache:
                matched_len, _ = await self.context_client.match_prefix(
                    tokens, session_id=session_id
                )
                if matched_len == prompt_len:
                    cache_hit = True

            if cache_hit:
                last_tok = input_ids[:, -1:]
                pos_tensor = torch.tensor([[prompt_len - 1]], device=input_ids.device)
                with self._forward_lock, torch.no_grad():
                    self.set_active_session(session_id)
                    out_fast = self.model(
                        last_tok, position_ids=pos_tensor, use_cache=False
                    )
                    next_tok = self._sample_next_token(
                        out_fast.logits, temperature=temperature, top_p=top_p
                    )
            elif 0 < matched_len < prompt_len:
                suffix_ids = input_ids[:, matched_len:]
                suffix_pos = torch.arange(
                    matched_len, prompt_len, device=input_ids.device
                ).unsqueeze(0)
                suffix_cache = DynamicCache()
                with self._forward_lock, torch.no_grad():
                    self.set_active_session(session_id, context_len=matched_len)
                    out_suffix = self.model(
                        suffix_ids,
                        past_key_values=suffix_cache,
                        position_ids=suffix_pos,
                        use_cache=True,
                    )
                    next_tok = self._sample_next_token(
                        out_suffix.logits, temperature=temperature, top_p=top_p
                    )

                suffix_tokens = tokens[matched_len:]
                num_layers = len(suffix_cache.layers)
                upload_id = uuid.uuid4().hex
                tasks = [
                    self.context_client.append_kv(
                        session_id=session_id,
                        layer_idx=l_idx,
                        k_delta=layer.keys,
                        v_delta=layer.values,
                        token_ids=suffix_tokens if l_idx == num_layers - 1 else None,
                        num_layers=num_layers,
                        upload_id=upload_id,
                    )
                    for l_idx, layer in enumerate(suffix_cache.layers)
                ]
                await asyncio.gather(*tasks)
                self.set_active_session(session_id, context_len=None)
            else:
                next_tok, prompt_len = await self.prefill_and_offload_async(
                    input_ids,
                    session_id,
                    temperature=temperature,
                    top_p=top_p,
                )

            first_tok_id = int(next_tok.item())
            if first_tok_id in eos_ids:
                self._set_finish_reason("stop")
                yield first_tok_id
                return

            yield first_tok_id
            cur_tok = next_tok
            local_cache = DynamicCache()

            for step in range(1, max_new_tokens):
                position = prompt_len + step - 1
                pos_tensor = torch.tensor([[position]], device=input_ids.device)

                with self._forward_lock, torch.no_grad():
                    self.set_active_session(session_id, context_len=None)
                    out = self.model(
                        cur_tok,
                        past_key_values=local_cache,
                        use_cache=True,
                        position_ids=pos_tensor,
                    )
                    cur_tok = self._sample_next_token(
                        out.logits, temperature=temperature, top_p=top_p
                    )
                tok_id = int(cur_tok.item())
                if tok_id in eos_ids:
                    self._set_finish_reason("stop")
                    yield tok_id
                    return
                yield tok_id

        finally:
            if release_on_completion:
                with contextlib.suppress(Exception):
                    await self.context_client.release_session(session_id)
            self.set_active_session(None)

    async def generate_async(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        session_id: str = "default-session",
        use_prefix_cache: bool = True,
        release_on_completion: bool = True,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[int]:
        """Asynchronously generates tokens using disaggregated attention merging.

        Args:
            input_ids: Prompt token tensor of shape `[1, prompt_len]`.
            max_new_tokens: Maximum number of new tokens to generate.
            session_id: Session identifier.
            use_prefix_cache: If True, checks for existing prefix cache on Context Server.
            release_on_completion: If True, releases session KV cache on completion.
            temperature: Sampling temperature (0.0 for greedy decoding).
            top_p: Nucleus sampling probability threshold in (0.0, 1.0].

        Returns:
            List of generated token IDs (including the first generated token).
        """
        generated: list[int] = []
        async for tok in self.generate_stream_async(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            session_id=session_id,
            use_prefix_cache=use_prefix_cache,
            release_on_completion=release_on_completion,
            temperature=temperature,
            top_p=top_p,
        ):
            generated.append(tok)
        return generated


__all__ = [
    "DisaggregatedAttentionPatcher",
    "DisaggregatedModel",
]
