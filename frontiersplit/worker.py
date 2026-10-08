"""FrontierSplit Pipeline Worker.

Runs on each compute node in the cluster, hosting assigned transformer layers.
Receives activation packets, computes local layer math, and forwards the resulting
tensor to the next node in the pipeline chain. Supports both real Hugging Face PyTorch
transformer layers and lightweight synthetic layers for testing.
"""

from __future__ import annotations

import argparse
import gc
import time
from typing import Any, List, Optional
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
import uvicorn

from frontiersplit.models import resolve_model_spec
from frontiersplit.protocol import (
    ActivationPacket,
    BatchedActivationPacket,
    BatchedGenerationResponse,
    GenerationResponse,
    ReleaseSessionPacket,
    ReleaseSessionResponse,
)

# Optional PyTorch and Hugging Face imports
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
    except ImportError:
        def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            def rotate_half(x):
                x1 = x[..., : x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1)
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            return q_embed, k_embed
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class WorkerKVCacheStore:
    """Manages session-keyed KV caches and VRAM budget guard for a pipeline worker stage."""

    def __init__(self, max_tokens_budget: int = 100_000):
        self.max_tokens_budget = max_tokens_budget
        self.allocated_tokens: int = 0
        self.sessions: Dict[str, Any] = {}
        self.session_caps: Dict[str, int] = {}
        self.session_seq_lens: Dict[str, int] = {}

    def get_or_create(self, request_id: str, prompt_len: int, max_tokens: Optional[int] = None) -> Any:
        """Retrieve existing session cache or allocate a new tile-aligned capacity."""
        if request_id in self.sessions:
            return self.sessions[request_id]

        hard_cap = prompt_len + (max_tokens if max_tokens is not None else 512)
        aligned_cap = ((hard_cap + 15) // 16) * 16

        if HAS_TORCH:
            try:
                from transformers.cache_utils import DynamicCache
                cache = DynamicCache()
            except ImportError:
                cache = None
        else:
            cache = []

        self.sessions[request_id] = cache
        self.session_caps[request_id] = aligned_cap
        self.session_seq_lens[request_id] = prompt_len
        self.allocated_tokens += aligned_cap
        return cache

    def get(self, request_id: str) -> Optional[Any]:
        return self.sessions.get(request_id)

    def update_seq_len(self, request_id: str, delta: int = 1) -> int:
        curr = self.session_seq_lens.get(request_id, 0) + delta
        self.session_seq_lens[request_id] = curr
        return curr

    def get_seq_len(self, request_id: str) -> int:
        return self.session_seq_lens.get(request_id, 0)

    def release(self, request_id: str) -> bool:
        if request_id in self.sessions:
            self.sessions.pop(request_id, None)
            cap = self.session_caps.pop(request_id, 0)
            self.session_seq_lens.pop(request_id, None)
            self.allocated_tokens = max(0, self.allocated_tokens - cap)
            return True
        return False

    def clear(self) -> None:
        self.sessions.clear()
        self.session_caps.clear()
        self.session_seq_lens.clear()
        self.allocated_tokens = 0



def create_worker_app(
    stage_id: int,
    total_stages: int,
    downstream_url: Optional[str] = None,
    model_name_or_path: Optional[str] = None,
    device: Optional[str] = None,
    hidden_size: int = 4096,
    vocab_size: int = 32000,
) -> FastAPI:
    """Creates a FastAPI app representing a single pipeline stage worker."""
    app = FastAPI(title=f"FrontierSplit-Worker-Stage-{stage_id}")
    is_first_stage = (stage_id == 0)
    is_final_stage = (stage_id == total_stages - 1)

    use_real_model = bool(model_name_or_path and HAS_TORCH)
    tokenizer = None
    assigned_layers: List[Any] = []
    embed_tokens = None
    final_norm = None
    lm_head = None
    layer_range = (0, 0)
    resolved_device = "cpu"

    if use_real_model:
        resolved_device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        print(f"[Worker Stage {stage_id}] Loading real model weights: {model_name_or_path} on device: {resolved_device}...")
        
        config = AutoConfig.from_pretrained(model_name_or_path)
        total_layers = getattr(config, "num_hidden_layers", 32)
        layers_per_stage = total_layers // total_stages
        remainder = total_layers % total_stages

        start_layer = stage_id * layers_per_stage + min(stage_id, remainder)
        assigned_count = layers_per_stage + (1 if stage_id < remainder else 0)
        end_layer = start_layer + assigned_count - 1
        layer_range = (start_layer, end_layer)
        print(f"[Worker Stage {stage_id}] Assigned layers {start_layer} to {end_layer} ({assigned_count} layers total).")

        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        except Exception as e:
            print(f"[Worker Stage {stage_id}] Warning: Failed to load tokenizer: {e}")

        # Resolve architecture specification and stop tokens dynamically for this model
        model_spec = resolve_model_spec(model_name_or_path, tokenizer=tokenizer)
        print(f"[Worker Stage {stage_id}] Model spec resolved: {model_spec.architecture}, context={model_spec.context_window}, stop_tokens={len(model_spec.stop_token_ids)}")


        # Load model weights in FP16
        full_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        model_config = getattr(full_model, "config", None)
        mask_function = None
        try:
            from transformers.models.mistral.modeling_mistral import create_causal_mask, create_sliding_window_causal_mask
            mask_function = create_causal_mask if getattr(model_config, "sliding_window", None) is None else create_sliding_window_causal_mask
        except Exception as e:
            print(f"[Worker Stage {stage_id}] Warning: Could not import Mistral causal mask functions: {e}")

        base_model = getattr(full_model, "model", getattr(full_model, "transformer", full_model))
        all_layers = getattr(base_model, "layers", getattr(base_model, "h", []))
        raw_rotary = getattr(base_model, "rotary_emb", None)
        rotary_emb = raw_rotary.to(resolved_device) if raw_rotary is not None else None

        if is_first_stage:
            raw_embed = getattr(base_model, "embed_tokens", getattr(base_model, "wte", None))
            embed_tokens = raw_embed.to(resolved_device) if raw_embed is not None else None

        for idx in range(start_layer, end_layer + 1):
            layer = all_layers[idx].to(resolved_device)
            # Re-index layer's internal self_attn.layer_idx to local index (0..len-1)
            # so that each stage worker tracks KV cache cleanly without empty padding
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = idx - start_layer
            assigned_layers.append(layer)

        if is_final_stage:
            raw_norm = getattr(base_model, "norm", getattr(base_model, "ln_f", None))
            final_norm = raw_norm.to(resolved_device) if raw_norm is not None else None
            raw_lm_head = getattr(full_model, "lm_head", None)
            lm_head = raw_lm_head.to(resolved_device) if raw_lm_head is not None else None

        # Free unassigned layers and garbage collect
        del full_model
        del base_model
        del all_layers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[Worker Stage {stage_id}] Successfully mounted assigned layers into VRAM.")

    else:
        # Synthetic fallback mode for testing and development
        np.random.seed(42 + stage_id)
        layer_proj = np.eye(hidden_size, dtype=np.float32) + 0.001 * np.random.randn(hidden_size, hidden_size).astype(np.float32)
        embeddings = np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.02 if is_first_stage else None
        lm_head_synth = np.random.randn(hidden_size, vocab_size).astype(np.float32) * 0.02 if is_final_stage else None

    kv_store = WorkerKVCacheStore()

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "stage_id": stage_id,
            "total_stages": total_stages,
            "is_first_stage": is_first_stage,
            "is_final_stage": is_final_stage,
            "downstream_url": downstream_url,
            "use_real_model": use_real_model,
            "device": resolved_device,
            "layer_range": layer_range,
            "active_sessions": len(kv_store.sessions),
            "allocated_kv_tokens": kv_store.allocated_tokens,
        }

    @app.post("/release_sessions", response_model=ReleaseSessionResponse)
    def release_sessions(packet: ReleaseSessionPacket):
        released = 0
        for req_id in packet.request_ids:
            if kv_store.release(req_id):
                released += 1

        if not is_final_stage and downstream_url:
            try:
                requests.post(f"{downstream_url}/release_sessions", json=packet.model_dump(), timeout=10)
            except Exception as e:
                print(f"[Worker Stage {stage_id}] Warning: Propagating release_sessions to {downstream_url} failed: {e}")

        return ReleaseSessionResponse(
            status="ok",
            released_count=released,
            active_sessions=len(kv_store.sessions),
        )

    @app.post("/forward", response_model=GenerationResponse)
    def forward(packet: ActivationPacket):
        start_time = time.time()

        if use_real_model:
            with torch.no_grad():
                if is_first_stage and packet.tokens:
                    token_tensor = torch.tensor([packet.tokens], device=resolved_device, dtype=torch.long)
                    hidden_states = embed_tokens(token_tensor)
                else:
                    arr = packet.get_tensor()
                    hidden_states = torch.from_numpy(arr).to(device=resolved_device, dtype=torch.float16)

                if packet.use_kv_cache:
                    if packet.is_prefill:
                        prompt_len = hidden_states.shape[1]
                        cache = kv_store.get_or_create(packet.request_id, prompt_len=prompt_len, max_tokens=packet.max_tokens)
                        pos_ids = torch.arange(prompt_len, dtype=torch.long, device=resolved_device).unsqueeze(0)
                        causal_mask = mask_function(
                            config=model_config,
                            inputs_embeds=hidden_states,
                            attention_mask=None,
                            past_key_values=cache,
                            position_ids=pos_ids,
                        ) if mask_function is not None else None
                        pos_emb = rotary_emb(hidden_states, position_ids=pos_ids) if rotary_emb is not None else None
                        for layer in assigned_layers:
                            layer_out = layer(
                                hidden_states,
                                attention_mask=causal_mask,
                                position_ids=pos_ids,
                                past_key_values=cache,
                                use_cache=True,
                                position_embeddings=pos_emb,
                            )
                            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                    else:
                        cache = kv_store.get(packet.request_id)
                        if cache is not None:
                            curr_len = cache.get_seq_length()
                            pos_ids = torch.tensor([[curr_len]], dtype=torch.long, device=resolved_device)
                            causal_mask = mask_function(
                                config=model_config,
                                inputs_embeds=hidden_states,
                                attention_mask=None,
                                past_key_values=cache,
                                position_ids=pos_ids,
                            ) if mask_function is not None else None
                            pos_emb = rotary_emb(hidden_states, position_ids=pos_ids) if rotary_emb is not None else None
                            for layer in assigned_layers:
                                layer_out = layer(
                                    hidden_states,
                                    attention_mask=causal_mask,
                                    position_ids=pos_ids,
                                    past_key_values=cache,
                                    use_cache=True,
                                    position_embeddings=pos_emb,
                                )
                                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                            kv_store.update_seq_len(packet.request_id, 1)
                        else:
                            seq_len = hidden_states.shape[1]
                            pos_ids = torch.arange(seq_len, dtype=torch.long, device=resolved_device).unsqueeze(0)
                            causal_mask = mask_function(
                                config=model_config,
                                inputs_embeds=hidden_states,
                                attention_mask=None,
                                past_key_values=None,
                                position_ids=pos_ids,
                            ) if mask_function is not None else None
                            pos_emb = rotary_emb(hidden_states, position_ids=pos_ids) if rotary_emb is not None else None
                            for layer in assigned_layers:
                                layer_out = layer(
                                    hidden_states,
                                    attention_mask=causal_mask,
                                    position_ids=pos_ids,
                                    position_embeddings=pos_emb,
                                )
                                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                else:
                    seq_len = hidden_states.shape[1]
                    pos_ids = torch.arange(seq_len, dtype=torch.long, device=resolved_device).unsqueeze(0)
                    causal_mask = mask_function(
                        config=model_config,
                        inputs_embeds=hidden_states,
                        attention_mask=None,
                        past_key_values=None,
                        position_ids=pos_ids,
                    ) if mask_function is not None else None
                    pos_emb = rotary_emb(hidden_states, position_ids=pos_ids) if rotary_emb is not None else None
                    for layer in assigned_layers:
                        layer_out = layer(
                            hidden_states,
                            attention_mask=causal_mask,
                            position_ids=pos_ids,
                            position_embeddings=pos_emb,
                        )
                        hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

                stage_compute_ms = (time.time() - start_time) * 1000
                timings = dict(packet.stage_timings)
                timings[f"stage_{stage_id}_compute_ms"] = stage_compute_ms

                if not is_final_stage:
                    if not downstream_url:
                        raise HTTPException(status_code=500, detail="Missing downstream_url for intermediate stage")

                    next_act = hidden_states.detach().cpu().to(torch.float32).numpy()
                    next_packet = ActivationPacket(
                        request_id=packet.request_id,
                        sequence_step=packet.sequence_step,
                        stage_id=stage_id + 1,
                        is_prefill=packet.is_prefill,
                        use_kv_cache=packet.use_kv_cache,
                        max_tokens=packet.max_tokens,
                        stage_timings=timings,
                    )
                    next_packet.set_tensor(next_act)

                    try:
                        resp = requests.post(f"{downstream_url}/forward", json=next_packet.model_dump(), timeout=600)
                        resp.raise_for_status()
                        return resp.json()
                    except Exception as e:
                        raise HTTPException(status_code=502, detail=f"Downstream handoff to {downstream_url} failed: {e}")

                else:
                    if final_norm is not None:
                        hidden_states = final_norm(hidden_states)

                    last_token_hidden = hidden_states[:, -1, :]  # shape: [1, hidden_size]
                    logits = lm_head(last_token_hidden)  # shape: [1, vocab_size]

                    next_token_id = int(torch.argmax(logits, dim=-1).item())
                    elapsed_ms = (time.time() - start_time) * 1000

                    if tokenizer is not None:
                        decoded_text = tokenizer.decode([next_token_id])
                        is_finished = (next_token_id in model_spec.stop_token_ids) or (packet.sequence_step >= model_spec.context_window)
                    else:
                        decoded_text = f"tok_{next_token_id % 100} "
                        is_finished = False

                    return GenerationResponse(
                        request_id=packet.request_id,
                        token_id=next_token_id,
                        text=decoded_text,
                        is_finished=is_finished,
                        latency_ms=elapsed_ms,
                        stage_timings=timings,
                    )

        else:
            # Synthetic matrix compute for testing
            if is_first_stage and packet.tokens:
                token_ids = packet.tokens
                activation = embeddings[token_ids]
                if activation.ndim == 2:
                    activation = activation[np.newaxis, ...]
            else:
                activation = packet.get_tensor()

            if packet.use_kv_cache:
                if packet.is_prefill:
                    kv_store.get_or_create(packet.request_id, prompt_len=len(packet.tokens or [1]), max_tokens=packet.max_tokens)
                else:
                    kv_store.update_seq_len(packet.request_id, 1)

            activation = np.matmul(activation, layer_proj)
            stage_compute_ms = (time.time() - start_time) * 1000
            timings = dict(packet.stage_timings)
            timings[f"stage_{stage_id}_compute_ms"] = stage_compute_ms

            if not is_final_stage:
                if not downstream_url:
                    raise HTTPException(status_code=500, detail="Missing downstream_url for intermediate stage")

                next_packet = ActivationPacket(
                    request_id=packet.request_id,
                    sequence_step=packet.sequence_step,
                    stage_id=stage_id + 1,
                    is_prefill=packet.is_prefill,
                    use_kv_cache=packet.use_kv_cache,
                    max_tokens=packet.max_tokens,
                    stage_timings=timings,
                )
                next_packet.set_tensor(activation)

                try:
                    resp = requests.post(f"{downstream_url}/forward", json=next_packet.model_dump(), timeout=30)
                    resp.raise_for_status()
                    return resp.json()
                except Exception as e:
                    raise HTTPException(status_code=502, detail=f"Downstream handoff to {downstream_url} failed: {e}")

            else:
                last_token_hidden = activation[0, -1, :]
                logits = np.matmul(last_token_hidden, lm_head_synth)
                next_token_id = int(np.argmax(logits))
                elapsed_ms = (time.time() - start_time) * 1000
                mock_word = f"tok_{next_token_id % 100} "
                return GenerationResponse(
                    request_id=packet.request_id,
                    token_id=next_token_id,
                    text=mock_word,
                    is_finished=(packet.sequence_step >= 32),
                    latency_ms=elapsed_ms,
                    stage_timings=timings,
                )

    @app.post("/forward_batched", response_model=BatchedGenerationResponse)
    def forward_batched(packet: BatchedActivationPacket):
        start_time = time.time()
        batch_size = len(packet.request_ids)

        if use_real_model:
            with torch.no_grad():
                if is_first_stage and packet.tokens_batch:
                    token_tensor = torch.tensor(packet.tokens_batch, device=resolved_device, dtype=torch.long)
                    hidden_states = embed_tokens(token_tensor)
                else:
                    arr = packet.get_tensor()
                    hidden_states = torch.from_numpy(arr).to(device=resolved_device, dtype=torch.float16)

                if packet.use_kv_cache and not packet.is_prefill:
                    # True Batched GEMM Decode:
                    # Linear projections and MLPs are executed across the full batch [B, 1, 4096] in ONE GEMM pass,
                    # reading the 7.5 GB of layer weights exactly once rather than B times.
                    B_cur = hidden_states.shape[0]
                    pos_ids = torch.tensor(
                        [[kv_store.get(req_id).get_seq_length() if kv_store.get(req_id) else 0] for req_id in packet.request_ids],
                        dtype=torch.long,
                        device=resolved_device,
                    )

                    for layer in assigned_layers:
                        residual = hidden_states
                        hidden_norm = layer.input_layernorm(hidden_states)

                        input_shape = hidden_norm.shape[:-1]
                        num_q_heads = layer.self_attn.config.num_attention_heads
                        num_kv_heads = layer.self_attn.config.num_key_value_heads
                        head_dim = layer.self_attn.head_dim

                        # Batched Q, K, V projections [B, 1, 4096] in a single GEMM
                        q = layer.self_attn.q_proj(hidden_norm).view(*input_shape, num_q_heads, head_dim).transpose(1, 2)
                        k = layer.self_attn.k_proj(hidden_norm).view(*input_shape, num_kv_heads, head_dim).transpose(1, 2)
                        v = layer.self_attn.v_proj(hidden_norm).view(*input_shape, num_kv_heads, head_dim).transpose(1, 2)

                        cos, sin = rotary_emb(hidden_norm, pos_ids)
                        q, k = apply_rotary_pos_emb(q, k, cos, sin)

                        # Per-stream attention using each session's contiguous DynamicCache
                        attn_outs = []
                        num_kv_groups = getattr(layer.self_attn, "num_key_value_groups", 1)
                        for b_idx, req_id in enumerate(packet.request_ids):
                            cache = kv_store.get(req_id)
                            q_b = q[b_idx:b_idx+1]
                            k_b = k[b_idx:b_idx+1]
                            v_b = v[b_idx:b_idx+1]
                            if cache is not None:
                                k_cached, v_cached = cache.update(k_b, v_b, layer.self_attn.layer_idx)
                            else:
                                k_cached, v_cached = k_b, v_b
                            if num_kv_groups > 1:
                                k_cached = k_cached.repeat_interleave(num_kv_groups, dim=1)
                                v_cached = v_cached.repeat_interleave(num_kv_groups, dim=1)
                            out_b = F.scaled_dot_product_attention(q_b, k_cached, v_cached)
                            attn_outs.append(out_b)

                        attn_out = torch.cat(attn_outs, dim=0).transpose(1, 2).reshape(B_cur, 1, -1)
                        hidden_states = residual + layer.self_attn.o_proj(attn_out)

                        # Batched Feed-Forward / MoE: [B, 1, 4096] in ONE GEMM pass!
                        normed = layer.post_attention_layernorm(hidden_states)
                        if hasattr(layer, "block_sparse_moe"):
                            moe_out = layer.block_sparse_moe(normed)
                            moe_out = moe_out[0] if isinstance(moe_out, tuple) else moe_out
                            hidden_states = hidden_states + moe_out
                        elif hasattr(layer, "mlp"):
                            hidden_states = hidden_states + layer.mlp(normed)
                        else:
                            raise RuntimeError(f"Unknown layer feed-forward block: {type(layer)}")

                    for req_id in packet.request_ids:
                        kv_store.update_seq_len(req_id, 1)

                elif packet.use_kv_cache and packet.is_prefill:
                    # Batched prefill with session KV caches
                    batch_outs = []
                    for b_idx, req_id in enumerate(packet.request_ids):
                        h_b = hidden_states[b_idx:b_idx+1, :, :]
                        prompt_len = h_b.shape[1]
                        max_tok = packet.max_tokens_list[b_idx] if packet.max_tokens_list else None
                        cache = kv_store.get_or_create(req_id, prompt_len=prompt_len, max_tokens=max_tok)
                        pos_ids = torch.arange(prompt_len, dtype=torch.long, device=resolved_device).unsqueeze(0)
                        c_mask = mask_function(
                            config=model_config,
                            inputs_embeds=h_b,
                            attention_mask=None,
                            past_key_values=cache,
                            position_ids=pos_ids,
                        ) if mask_function is not None else None
                        p_emb = rotary_emb(h_b, position_ids=pos_ids) if rotary_emb is not None else None
                        for layer in assigned_layers:
                            l_out = layer(
                                h_b,
                                attention_mask=c_mask,
                                position_ids=pos_ids,
                                past_key_values=cache,
                                use_cache=True,
                                position_embeddings=p_emb,
                            )
                            h_b = l_out[0] if isinstance(l_out, tuple) else l_out
                        batch_outs.append(h_b)
                    hidden_states = torch.cat(batch_outs, dim=0)

                else:
                    seq_len = hidden_states.shape[1]
                    mask_tensor = None
                    if packet.attention_mask:
                        mask_tensor = torch.tensor(packet.attention_mask, device=resolved_device, dtype=torch.long)
                        position_ids = (mask_tensor.cumsum(dim=-1) - 1).clamp(min=0)
                    else:
                        position_ids = torch.arange(seq_len, dtype=torch.long, device=resolved_device).unsqueeze(0).expand(batch_size, -1)

                    c_mask = mask_function(
                        config=model_config,
                        inputs_embeds=hidden_states,
                        attention_mask=mask_tensor,
                        past_key_values=None,
                        position_ids=position_ids,
                    ) if mask_function is not None else None
                    pos_emb = rotary_emb(hidden_states, position_ids=position_ids) if rotary_emb is not None else None

                    for layer in assigned_layers:
                        l_out = layer(
                            hidden_states,
                            attention_mask=c_mask,
                            position_ids=position_ids,
                            position_embeddings=pos_emb,
                        )
                        hidden_states = l_out[0] if isinstance(l_out, tuple) else l_out

                stage_compute_ms = (time.time() - start_time) * 1000
                timings = dict(packet.stage_timings)
                timings[f"stage_{stage_id}_compute_ms"] = stage_compute_ms

                if not is_final_stage:
                    if not downstream_url:
                        raise HTTPException(status_code=500, detail="Missing downstream_url for intermediate stage")

                    next_act = hidden_states.detach().cpu().to(torch.float32).numpy()
                    next_packet = BatchedActivationPacket(
                        request_ids=packet.request_ids,
                        sequence_steps=packet.sequence_steps,
                        stage_id=stage_id + 1,
                        is_prefill=packet.is_prefill,
                        use_kv_cache=packet.use_kv_cache,
                        max_tokens_list=packet.max_tokens_list,
                        attention_mask=packet.attention_mask,
                        stage_timings=timings,
                    )
                    next_packet.set_tensor(next_act)

                    try:
                        resp = requests.post(f"{downstream_url}/forward_batched", json=next_packet.model_dump(), timeout=600)
                        resp.raise_for_status()
                        return resp.json()
                    except Exception as e:
                        raise HTTPException(status_code=502, detail=f"Downstream batched handoff to {downstream_url} failed: {e}")

                else:
                    if final_norm is not None:
                        hidden_states = final_norm(hidden_states)

                    last_token_hidden = hidden_states[:, -1, :]  # shape: [B, hidden_size]
                    logits = lm_head(last_token_hidden)  # shape: [B, vocab_size]

                    next_token_ids = torch.argmax(logits, dim=-1).tolist()
                    elapsed_ms = (time.time() - start_time) * 1000

                    responses = []
                    for req_id, seq_step, next_tok in zip(packet.request_ids, packet.sequence_steps, next_token_ids):
                        if tokenizer is not None:
                            decoded_text = tokenizer.decode([next_tok])
                            is_finished = (next_tok in model_spec.stop_token_ids) or (seq_step >= model_spec.context_window)
                        else:
                            decoded_text = f"tok_{next_tok % 100} "
                            is_finished = False

                        responses.append(
                            GenerationResponse(
                                request_id=req_id,
                                token_id=next_tok,
                                text=decoded_text,
                                is_finished=is_finished,
                                latency_ms=elapsed_ms,
                                stage_timings=timings,
                            )
                        )

                    return BatchedGenerationResponse(
                        responses=responses,
                        batch_size=len(responses),
                        stage_timings=timings,
                    )

        else:
            # Synthetic batched matrix compute
            if is_first_stage and packet.tokens_batch:
                max_tok_len = max(len(t) for t in packet.tokens_batch)
                padded_batch = [t + [0] * (max_tok_len - len(t)) for t in packet.tokens_batch]
                tokens_arr = np.array(padded_batch)
                activation = embeddings[tokens_arr]
            else:
                activation = packet.get_tensor()

            if packet.use_kv_cache:
                for b_idx, req_id in enumerate(packet.request_ids):
                    if packet.is_prefill:
                        max_tok = packet.max_tokens_list[b_idx] if packet.max_tokens_list else None
                        prompt_len = len(packet.tokens_batch[b_idx]) if (packet.tokens_batch and b_idx < len(packet.tokens_batch)) else 1
                        kv_store.get_or_create(req_id, prompt_len=prompt_len, max_tokens=max_tok)
                    else:
                        kv_store.update_seq_len(req_id, 1)

            activation = np.matmul(activation, layer_proj)
            stage_compute_ms = (time.time() - start_time) * 1000
            timings = dict(packet.stage_timings)
            timings[f"stage_{stage_id}_compute_ms"] = stage_compute_ms

            if not is_final_stage:
                if not downstream_url:
                    raise HTTPException(status_code=500, detail="Missing downstream_url for intermediate stage")

                next_packet = BatchedActivationPacket(
                    request_ids=packet.request_ids,
                    sequence_steps=packet.sequence_steps,
                    stage_id=stage_id + 1,
                    is_prefill=packet.is_prefill,
                    use_kv_cache=packet.use_kv_cache,
                    max_tokens_list=packet.max_tokens_list,
                    attention_mask=packet.attention_mask,
                    stage_timings=timings,
                )
                next_packet.set_tensor(activation)

                try:
                    resp = requests.post(f"{downstream_url}/forward_batched", json=next_packet.model_dump(), timeout=30)
                    resp.raise_for_status()
                    return resp.json()
                except Exception as e:
                    raise HTTPException(status_code=502, detail=f"Downstream batched handoff to {downstream_url} failed: {e}")

            else:
                last_token_hidden = activation[:, -1, :]
                logits = np.matmul(last_token_hidden, lm_head_synth)
                next_token_ids = np.argmax(logits, axis=-1).tolist()
                elapsed_ms = (time.time() - start_time) * 1000

                responses = [
                    GenerationResponse(
                        request_id=req_id,
                        token_id=tok,
                        text=f"tok_{tok % 100} ",
                        is_finished=(seq_step >= 32),
                        latency_ms=elapsed_ms,
                        stage_timings=timings,
                    )
                    for req_id, seq_step, tok in zip(packet.request_ids, packet.sequence_steps, next_token_ids)
                ]
                return BatchedGenerationResponse(
                    responses=responses,
                    batch_size=len(responses),
                    stage_timings=timings,
                )

    return app


def main():
    parser = argparse.ArgumentParser(description="FrontierSplit Pipeline Stage Worker")
    parser.add_argument("--stage-id", type=int, required=True, help="Stage index (0 to total_stages - 1)")
    parser.add_argument("--total-stages", type=int, default=2, help="Total pipeline stages")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on")
    parser.add_argument("--downstream-url", type=str, default=None, help="Downstream worker URL")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    parser.add_argument("--model-name", type=str, default=None, help="Hugging Face model ID to load")
    parser.add_argument("--device", type=str, default=None, help="PyTorch device (cuda, mps, cpu)")
    args = parser.parse_args()

    app = create_worker_app(
        stage_id=args.stage_id,
        total_stages=args.total_stages,
        downstream_url=args.downstream_url,
        model_name_or_path=args.model_name,
        device=args.device,
    )
    print(f"Starting FrontierSplit Worker Stage {args.stage_id}/{args.total_stages} on port {args.port}...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
