"""HuggingFace Model Integration for FrontierSplit Disaggregated Inference.

Wraps any standard HuggingFace autoregressive causal language model (Llama,
Mistral, Qwen2, Gemma) to offload prompt KV caches to a remote Context Server
and generate tokens using Online Softmax attention merging.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
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

try:
    import anyio
except ImportError:
    anyio = None

logger = logging.getLogger("frontiersplit.hf_model")

_active_context_client_var: contextvars.ContextVar[ContextClient | None] = (
    contextvars.ContextVar("frontiersplit_active_context_client", default=None)
)


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

    @property
    def client(self) -> ContextClient:
        """Returns the task-bound ContextClient if set, else fallback to instance client."""
        c = _active_context_client_var.get()
        return c if c is not None else self.context_client

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
        client = self.client
        scaling = getattr(self.original_attn, "scaling", 1.0 / (head_dim**0.5))
        sliding_window = self._get_sliding_window()

        if past_key_values is None:
            prompt_chunk = client.query_partial_attention_sync(
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
            is_causal = q_len > 1

            # Overlap remote TCP query with local partial attention compute
            sync_lock = getattr(client, "_sync_lock", contextlib.nullcontext())
            if hasattr(client, "send_query_partial_sync") and hasattr(
                client, "recv_query_partial_sync"
            ):
                with sync_lock:
                    try:
                        client.send_query_partial_sync(
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
                            q,
                            k,
                            v,
                            scale=scaling,
                            attention_mask=attention_mask,
                            is_causal=is_causal,
                            sliding_window=sliding_window,
                        )
                        prompt_chunk = client.recv_query_partial_sync(device=q.device)
                    except BaseException:
                        if hasattr(client, "close_sync"):
                            client.close_sync()
                        raise
            else:
                prompt_chunk = client.query_partial_attention_sync(
                    session_id=session_id,
                    layer_idx=self.layer_idx,
                    q=q,
                    scale=scaling,
                    context_len=context_len,
                    sliding_window=sliding_window,
                    past_local_len=kv_len,
                ).to(q.device)
                local_chunk = compute_partial_attention(
                    q,
                    k,
                    v,
                    scale=scaling,
                    attention_mask=attention_mask,
                    is_causal=is_causal,
                    sliding_window=sliding_window,
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
        self._active_sessions: set[str] = set()
        self._active_sessions_lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="frontiersplit-forward"
        )
        self._patch_layers()

    def __del__(self) -> None:
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

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

    @contextlib.contextmanager
    def _scoped_forward_context(
        self,
        session_id: str | None,
        context_len: int | None = None,
        client: ContextClient | None = None,
    ):
        """Scopes task-bound ContextClient and active session strictly around a model forward pass."""
        self.set_active_session(session_id, context_len)
        token = (
            _active_context_client_var.set(client)
            if client is not None
            else None
        )
        try:
            yield
        finally:
            if token is not None:
                _active_context_client_var.reset(token)
            self.set_active_session(None)

    def _get_context_client(self) -> ContextClient:
        """Returns the task-bound ContextClient if set, else fallback to instance client."""
        c = _active_context_client_var.get()
        return c if c is not None else self.context_client

    def prefill_and_offload(
        self,
        input_ids: torch.Tensor,
        session_id: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        context_client: ContextClient | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Runs prefill on the prompt and offloads static prompt KV caches to Context Server.

        Args:
            input_ids: Prompt token tensor of shape `[1, prompt_len]`.
            session_id: Unique session identifier.
            temperature: Sampling temperature (0.0 for greedy decoding).
            top_p: Nucleus sampling probability threshold in (0.0, 1.0].
            context_client: Optional request-bound ContextClient.

        Returns:
            Tuple of (first_generated_token_id, prompt_length).
        """
        self._validate_sampling_params(temperature, top_p)
        client = context_client or self._get_context_client()
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
        upload_id = f"prefill-{session_id}-0-{prompt_len}"

        # Offload each layer's static prompt KV cache to the remote Context Server
        for l_idx, layer in enumerate(prompt_cache.layers):
            k_prompt = layer.keys.clone()
            v_prompt = layer.values.clone()
            client.register_prompt_sync(
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
        reasoning_commit_threshold: int = 4000,
        context_client: ContextClient | None = None,
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
            reasoning_commit_threshold: Token count threshold above which generated reasoning
                tokens are committed to the Context Server (Review 4.3).
            context_client: Optional request-bound ContextClient.

        Yields:
            Generated token IDs one by one (including the first generated token).
        """
        self._validate_sampling_params(temperature, top_p)
        if max_new_tokens <= 0:
            self._set_finish_reason("length")
            return

        target_client = context_client or self.context_client
        tokens = input_ids[0].tolist()
        base_prompt_len = len(tokens)
        remote_prompt_len = base_prompt_len
        eos_ids = self._get_eos_token_ids()
        self._set_finish_reason("length")

        with self._active_sessions_lock:
            if session_id in self._active_sessions:
                raise RuntimeError(
                    f"Session '{session_id}' is already in use by an active generation. "
                    "Overlapping requests with the same session ID must be serialized."
                )
            self._active_sessions.add(session_id)

        # Initialize lifecycle state BEFORE try: so finally block is always safe
        generated_tokens: list[int] = []
        local_cache: DynamicCache | None = None
        last_committed_idx = 0

        try:
            # Check for warm prefix cache hit on Context Server
            cache_hit = False
            matched_len = 0
            if use_prefix_cache:
                matched_len, _ = target_client.match_prefix_sync(
                    tokens, session_id=session_id
                )
                if matched_len == base_prompt_len:
                    cache_hit = True

            if cache_hit:
                # 100% prefix cache hit: skip full prefill compute
                last_tok = input_ids[:, -1:]
                pos_tensor = torch.tensor(
                    [[base_prompt_len - 1]], device=input_ids.device
                )
                with (
                    self._forward_lock,
                    torch.no_grad(),
                    self._scoped_forward_context(session_id, None, target_client),
                ):
                    out_fast = self.model(
                        last_tok, position_ids=pos_tensor, use_cache=False
                    )
                    next_tok = self._sample_next_token(
                        out_fast.logits, temperature=temperature, top_p=top_p
                    )
            elif 0 < matched_len < base_prompt_len:
                # Disaggregated Delta Prefill:
                # Tokens [0..matched_len] are already warm on Context Server.
                # Suffix tokens [matched_len..prompt_len] run forward pass with disaggregated attention.
                suffix_ids = input_ids[:, matched_len:]
                suffix_pos = torch.arange(
                    matched_len, base_prompt_len, device=input_ids.device
                ).unsqueeze(0)
                suffix_cache = DynamicCache()
                with (
                    self._forward_lock,
                    torch.no_grad(),
                    self._scoped_forward_context(session_id, matched_len, target_client),
                ):
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
                upload_id = f"delta-{session_id}-{matched_len}-{base_prompt_len}"
                for l_idx, layer in enumerate(suffix_cache.layers):
                    target_client.append_kv_sync(
                        session_id=session_id,
                        layer_idx=l_idx,
                        k_delta=layer.keys,
                        v_delta=layer.values,
                        token_ids=suffix_tokens if l_idx == num_layers - 1 else None,
                        num_layers=num_layers,
                        upload_id=upload_id,
                    )
            else:
                # 1. Prefill and offload prompt KV
                next_tok, prompt_len = self.prefill_and_offload(
                    input_ids,
                    session_id,
                    temperature=temperature,
                    top_p=top_p,
                    context_client=target_client,
                )
                base_prompt_len = prompt_len
                remote_prompt_len = prompt_len

            first_tok_id = int(next_tok.item())
            generated_tokens.append(first_tok_id)
            local_cache = DynamicCache()

            if first_tok_id in eos_ids:
                self._set_finish_reason("stop")
                yield first_tok_id
                return

            yield first_tok_id
            cur_tok = next_tok

            # 3. Autoregressive decode loop
            for step in range(1, max_new_tokens):
                # Decode positions are strictly contiguous based on immutable base prompt length:
                position = base_prompt_len + step - 1
                pos_tensor = torch.tensor([[position]], device=input_ids.device)

                with (
                    self._forward_lock,
                    torch.no_grad(),
                    self._scoped_forward_context(session_id, None, target_client),
                ):
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
                generated_tokens.append(tok_id)

                # Mid-generation reasoning commit if output sequence exceeds reasoning_commit_threshold
                if (
                    reasoning_commit_threshold > 0
                    and len(local_cache.layers) > 0
                    and local_cache.layers[0].keys.shape[-2]
                    >= reasoning_commit_threshold
                ):
                    local_len = local_cache.layers[0].keys.shape[-2]
                    chunk_tokens = generated_tokens[
                        last_committed_idx : last_committed_idx + local_len
                    ]
                    num_layers = len(local_cache.layers)
                    upload_id = f"commit-{session_id}-{last_committed_idx}-{last_committed_idx + local_len}"
                    for l_idx, layer in enumerate(local_cache.layers):
                        target_client.append_kv_sync(
                            session_id=session_id,
                            layer_idx=l_idx,
                            k_delta=layer.keys,
                            v_delta=layer.values,
                            token_ids=chunk_tokens if l_idx == num_layers - 1 else None,
                            num_layers=num_layers,
                            upload_id=upload_id,
                        )
                    last_committed_idx += local_len
                    remote_prompt_len += local_len
                    local_cache = DynamicCache()

                if tok_id in eos_ids:
                    self._set_finish_reason("stop")
                    yield tok_id
                    return
                yield tok_id

        finally:
            try:
                if not release_on_completion:
                    # Turn-boundary reasoning trace commit policy (Review 4.3):
                    # If generated reasoning tokens reach threshold, commit them to the Context Server.
                    total_generated = len(generated_tokens)
                    if (
                        reasoning_commit_threshold > 0
                        and total_generated >= reasoning_commit_threshold
                        and local_cache is not None
                        and len(local_cache.layers) > 0
                        and local_cache.layers[0].keys.shape[-2] > 0
                    ):
                        rem_len = local_cache.layers[0].keys.shape[-2]
                        chunk_tokens = generated_tokens[
                            last_committed_idx : last_committed_idx + rem_len
                        ]
                        num_layers = len(local_cache.layers)
                        upload_id = f"commit-{session_id}-{last_committed_idx}-{last_committed_idx + rem_len}"
                        for l_idx, layer in enumerate(local_cache.layers):
                            target_client.append_kv_sync(
                                session_id=session_id,
                                layer_idx=l_idx,
                                k_delta=layer.keys,
                                v_delta=layer.values,
                                token_ids=chunk_tokens
                                if l_idx == num_layers - 1
                                else None,
                                num_layers=num_layers,
                                upload_id=upload_id,
                            )
                        last_committed_idx += rem_len
                        remote_prompt_len += rem_len

                if release_on_completion:
                    with contextlib.suppress(Exception):
                        target_client.release_session_sync(session_id)
            finally:
                with self._active_sessions_lock:
                    self._active_sessions.discard(session_id)
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
        reasoning_commit_threshold: int = 4000,
        context_client: ContextClient | None = None,
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
            reasoning_commit_threshold: Token count threshold above which generated reasoning
                tokens are committed to the Context Server (Review 4.3).
            context_client: Optional request-bound ContextClient.

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
                reasoning_commit_threshold=reasoning_commit_threshold,
                context_client=context_client,
            )
        )

    async def prefill_and_offload_async(
        self,
        input_ids: torch.Tensor,
        session_id: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        context_client: ContextClient | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Asynchronously runs prefill on the prompt and offloads KV caches."""
        self._validate_sampling_params(temperature, top_p)
        client = context_client or self.context_client
        prompt_cache = DynamicCache()

        def _run_prefill(ids_in, cache_in):
            with self._forward_lock, torch.no_grad():
                self.set_active_session(None)
                outputs = self.model(
                    ids_in, past_key_values=cache_in, use_cache=True
                )
                tok = self._sample_next_token(
                    outputs.logits, temperature=temperature, top_p=top_p
                )
                return outputs.past_key_values, tok

        active_prefill_fut = self._executor.submit(
            _run_prefill, input_ids, prompt_cache
        )
        try:
            prompt_cache, next_token = await asyncio.wrap_future(active_prefill_fut)
        finally:
            if not active_prefill_fut.done():
                with contextlib.suppress(BaseException):
                    active_prefill_fut.result()

        prompt_len = input_ids.shape[1]
        tokens = input_ids[0].tolist()
        num_layers = len(prompt_cache.layers)
        upload_id = f"prefill-{session_id}-0-{prompt_len}"

        tasks = []
        for l_idx, layer in enumerate(prompt_cache.layers):
            k_prompt = layer.keys.clone()
            v_prompt = layer.values.clone()
            tasks.append(
                client.register_prompt(
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
        reasoning_commit_threshold: int = 4000,
        context_client: ContextClient | None = None,
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
            reasoning_commit_threshold: Token count threshold above which generated reasoning
                tokens are committed to the Context Server (Review 4.3).
            context_client: Optional request-bound ContextClient.

        Yields:
            Generated token IDs one by one (including the first generated token).
        """
        self._validate_sampling_params(temperature, top_p)
        if max_new_tokens <= 0:
            self._set_finish_reason("length")
            return

        target_client = context_client or self.context_client
        tokens = input_ids[0].tolist()
        base_prompt_len = len(tokens)
        remote_prompt_len = base_prompt_len
        eos_ids = self._get_eos_token_ids()
        self._set_finish_reason("length")

        with self._active_sessions_lock:
            if session_id in self._active_sessions:
                raise RuntimeError(
                    f"Session '{session_id}' is already in use by an active generation. "
                    "Overlapping requests with the same session ID must be serialized."
                )
            self._active_sessions.add(session_id)

        # Initialize lifecycle state BEFORE try: so finally block is always safe
        generated_tokens: list[int] = []
        local_cache: DynamicCache | None = None
        last_committed_idx = 0
        active_forward_future: concurrent.futures.Future | None = None

        try:
            cache_hit = False
            matched_len = 0
            if use_prefix_cache:
                matched_len, _ = await target_client.match_prefix(
                    tokens, session_id=session_id
                )
                if matched_len == base_prompt_len:
                    cache_hit = True

            if cache_hit:
                last_tok = input_ids[:, -1:]
                pos_tensor = torch.tensor(
                    [[base_prompt_len - 1]], device=input_ids.device
                )

                def _run_fast_prefill(last_tok_in, pos_in):
                    with (
                        self._forward_lock,
                        torch.no_grad(),
                        self._scoped_forward_context(session_id, None, target_client),
                    ):
                        out_fast = self.model(
                            last_tok_in, position_ids=pos_in, use_cache=False
                        )
                        return self._sample_next_token(
                            out_fast.logits, temperature=temperature, top_p=top_p
                        )

                active_forward_future = self._executor.submit(
                    _run_fast_prefill, last_tok, pos_tensor
                )
                next_tok = await asyncio.wrap_future(active_forward_future)
                active_forward_future = None
            elif 0 < matched_len < base_prompt_len:
                suffix_ids = input_ids[:, matched_len:]
                suffix_pos = torch.arange(
                    matched_len, base_prompt_len, device=input_ids.device
                ).unsqueeze(0)
                suffix_cache = DynamicCache()

                def _run_delta_prefill(s_ids_in, s_pos_in, s_cache_in):
                    with (
                        self._forward_lock,
                        torch.no_grad(),
                        self._scoped_forward_context(
                            session_id, matched_len, target_client
                        ),
                    ):
                        out_suffix = self.model(
                            s_ids_in,
                            past_key_values=s_cache_in,
                            position_ids=s_pos_in,
                            use_cache=True,
                        )
                        return self._sample_next_token(
                            out_suffix.logits,
                            temperature=temperature,
                            top_p=top_p,
                        )

                active_forward_future = self._executor.submit(
                    _run_delta_prefill, suffix_ids, suffix_pos, suffix_cache
                )
                next_tok = await asyncio.wrap_future(active_forward_future)
                active_forward_future = None

                suffix_tokens = tokens[matched_len:]
                num_layers = len(suffix_cache.layers)
                upload_id = f"delta-{session_id}-{matched_len}-{base_prompt_len}"
                tasks = [
                    target_client.append_kv(
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
            else:
                next_tok, prompt_len = await self.prefill_and_offload_async(
                    input_ids,
                    session_id,
                    temperature=temperature,
                    top_p=top_p,
                    context_client=target_client,
                )
                base_prompt_len = prompt_len
                remote_prompt_len = prompt_len

            first_tok_id = int(next_tok.item())
            generated_tokens.append(first_tok_id)
            local_cache = DynamicCache()

            if first_tok_id in eos_ids:
                self._set_finish_reason("stop")
                yield first_tok_id
                return

            yield first_tok_id
            cur_tok = next_tok

            def _run_decode_step(cur_tok_in, pos_in, cache_in):
                with (
                    self._forward_lock,
                    torch.no_grad(),
                    self._scoped_forward_context(session_id, None, target_client),
                ):
                    out = self.model(
                        cur_tok_in,
                        past_key_values=cache_in,
                        use_cache=True,
                        position_ids=pos_in,
                    )
                    return self._sample_next_token(
                        out.logits, temperature=temperature, top_p=top_p
                    )

            for step in range(1, max_new_tokens):
                # Decode positions are strictly contiguous based on immutable base prompt length:
                position = base_prompt_len + step - 1
                pos_tensor = torch.tensor([[position]], device=input_ids.device)

                active_forward_future = self._executor.submit(
                    _run_decode_step, cur_tok, pos_tensor, local_cache
                )
                cur_tok = await asyncio.wrap_future(active_forward_future)
                active_forward_future = None
                tok_id = int(cur_tok.item())
                generated_tokens.append(tok_id)

                # Mid-generation reasoning commit if output sequence exceeds reasoning_commit_threshold
                if (
                    reasoning_commit_threshold > 0
                    and len(local_cache.layers) > 0
                    and local_cache.layers[0].keys.shape[-2]
                    >= reasoning_commit_threshold
                ):
                    local_len = local_cache.layers[0].keys.shape[-2]
                    chunk_tokens = generated_tokens[
                        last_committed_idx : last_committed_idx + local_len
                    ]
                    num_layers = len(local_cache.layers)
                    upload_id = f"commit-{session_id}-{last_committed_idx}-{last_committed_idx + local_len}"
                    tasks = [
                        target_client.append_kv(
                            session_id=session_id,
                            layer_idx=l_idx,
                            k_delta=layer.keys,
                            v_delta=layer.values,
                            token_ids=chunk_tokens
                            if l_idx == num_layers - 1
                            else None,
                            num_layers=num_layers,
                            upload_id=upload_id,
                        )
                        for l_idx, layer in enumerate(local_cache.layers)
                    ]
                    await asyncio.gather(*tasks)
                    last_committed_idx += local_len
                    remote_prompt_len += local_len
                    local_cache = DynamicCache()

                if tok_id in eos_ids:
                    self._set_finish_reason("stop")
                    yield tok_id
                    return
                yield tok_id

        finally:
            try:
                # 1. Drain worker thread synchronously so it finishes before any cleanup
                if (
                    active_forward_future is not None
                    and not active_forward_future.done()
                ):
                    with contextlib.suppress(BaseException):
                        active_forward_future.result()

                # 2. Shield async cleanup against AnyIO / Starlette cancel scopes (e.g. client disconnect)
                curr_task = asyncio.current_task()
                cancelling_count = 0
                if curr_task is not None and hasattr(curr_task, "uncancel"):
                    while curr_task.cancelling() > 0:
                        curr_task.uncancel()
                        cancelling_count += 1

                try:
                    shield_ctx = (
                        anyio.CancelScope(shield=True)
                        if anyio is not None
                        else contextlib.nullcontext()
                    )
                    with shield_ctx:
                        # 2a. Consistent cache commit for retained sessions
                        if not release_on_completion:
                            total_generated = len(generated_tokens)
                            completed_uncommitted = total_generated - last_committed_idx
                            if (
                                reasoning_commit_threshold > 0
                                and total_generated >= reasoning_commit_threshold
                                and local_cache is not None
                                and len(local_cache.layers) > 0
                                and completed_uncommitted > 0
                            ):
                                min_layer_len = min(
                                    layer.keys.shape[-2] for layer in local_cache.layers
                                )
                                valid_len = min(completed_uncommitted, min_layer_len)
                                if valid_len > 0:
                                    chunk_tokens = generated_tokens[
                                        last_committed_idx : last_committed_idx + valid_len
                                    ]
                                    num_layers = len(local_cache.layers)
                                    upload_id = f"commit-{session_id}-{last_committed_idx}-{last_committed_idx + valid_len}"
                                    tasks = [
                                        target_client.append_kv(
                                            session_id=session_id,
                                            layer_idx=l_idx,
                                            k_delta=layer.keys[..., :valid_len, :],
                                            v_delta=layer.values[..., :valid_len, :],
                                            token_ids=chunk_tokens
                                            if l_idx == num_layers - 1
                                            else None,
                                            num_layers=num_layers,
                                            upload_id=upload_id,
                                        )
                                        for l_idx, layer in enumerate(local_cache.layers)
                                    ]
                                    with contextlib.suppress(BaseException):
                                        await asyncio.gather(*tasks)

                        # 2b. Release session if requested
                        if release_on_completion:
                            with contextlib.suppress(BaseException):
                                await target_client.release_session(session_id)
                finally:
                    if curr_task is not None and hasattr(curr_task, "cancelling"):
                        for _ in range(cancelling_count):
                            curr_task.cancel()
            finally:
                with self._active_sessions_lock:
                    self._active_sessions.discard(session_id)
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
        reasoning_commit_threshold: int = 4000,
        context_client: ContextClient | None = None,
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
            reasoning_commit_threshold: Token count threshold above which generated reasoning
                tokens are committed to the Context Server (Review 4.3).
            context_client: Optional request-bound ContextClient.

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
            reasoning_commit_threshold=reasoning_commit_threshold,
            context_client=context_client,
        ):
            generated.append(tok)
        return generated


__all__ = [
    "DisaggregatedAttentionPatcher",
    "DisaggregatedModel",
]
