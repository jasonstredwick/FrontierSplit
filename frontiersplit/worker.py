"""FrontierSplit Pipeline Worker.

Runs on each compute node in the cluster, hosting assigned transformer layers.
Receives activation packets, computes local layer math, and forwards the resulting
tensor to the next node in the pipeline chain. Supports both real Hugging Face PyTorch
transformer layers and lightweight synthetic layers for testing.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import gc
import os
import time
import traceback
from typing import Any, Dict, List, Optional
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
import uvicorn

from frontiersplit.models import resolve_model_spec
from frontiersplit.protocol import (
    ActivationPacket,
    BatchedActivationPacket,
    BatchedGenerationResponse,
    ChunkActivationPacket,
    ChunkGenerationResponse,
    MSG_FORWARD_CHUNK_REQ,
    MSG_FORWARD_CHUNK_RESP,
    GenerationResponse,
    ReleaseSessionPacket,
    ReleaseSessionResponse,
)
from frontiersplit.transport import BinaryTransportClient, BinaryTransportServer

# Optional PyTorch and Hugging Face imports
try:
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        class _DummyStream:
            def wait_stream(self, *a, **kw): pass
        try:
            if hasattr(torch, "cuda"):
                torch.cuda.Stream = _DummyStream
        except Exception:
            pass
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


def _is_static_cache(cache: Any) -> bool:
    if cache is None:
        return False
    return (
        hasattr(cache, "get_max_length")
        or hasattr(cache, "get_max_cache_shape")
        or ("Static" in getattr(cache, "__class__", type).__name__)
        or ("Paged" in getattr(cache, "__class__", type).__name__)
    )


class PagedBlockPool:
    """Fixed-capacity physical block pool pre-allocated in TPU HBM to eliminate dynamic allocations."""

    def __init__(
        self,
        num_layers: int = 4,
        num_blocks: int = 2048,
        block_size: int = 16,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        device: str = "cpu",
        dtype: Any = None,
    ):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float16

        self.k_pool = torch.zeros(
            (num_layers, num_blocks, block_size, num_kv_heads, head_dim),
            dtype=self.dtype,
            device=device,
        )
        self.v_pool = torch.zeros(
            (num_layers, num_blocks, block_size, num_kv_heads, head_dim),
            dtype=self.dtype,
            device=device,
        )
        self.free_blocks: List[int] = list(range(num_blocks - 1, -1, -1))

    def allocate(self, count: int) -> List[int]:
        if len(self.free_blocks) < count:
            raise RuntimeError(
                f"PagedBlockPool out of blocks: requested {count}, free {len(self.free_blocks)}"
            )
        return [self.free_blocks.pop() for _ in range(count)]

    def free(self, blocks: List[int]) -> None:
        self.free_blocks.extend(blocks)


class PagedSessionCache:
    """Manages virtual-to-physical block mapping and contiguous cache views for a single request stream."""

    def __init__(self, pool: PagedBlockPool, initial_capacity: int = 64):
        self.pool = pool
        self.block_size = pool.block_size
        needed_blocks = max(1, (initial_capacity + self.block_size - 1) // self.block_size)
        self.block_table: List[int] = self.pool.allocate(needed_blocks)
        self._seen_tokens: int = 0
        self.capacity: int = len(self.block_table) * self.block_size

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen_tokens

    def get_max_length(self) -> int:
        return self.capacity

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h, num_new, d = key_states.shape
        start_pos = self._seen_tokens
        end_pos = start_pos + num_new

        if end_pos > self.capacity:
            needed = (end_pos - self.capacity + self.block_size - 1) // self.block_size
            if layer_idx == 0:
                self.block_table.extend(self.pool.allocate(needed))
                self.capacity = len(self.block_table) * self.block_size

        kt = key_states.squeeze(0).permute(1, 0, 2)
        vt = value_states.squeeze(0).permute(1, 0, 2)
        for i in range(num_new):
            pos = start_pos + i
            blk_idx = self.block_table[pos // self.block_size]
            offset = pos % self.block_size
            self.pool.k_pool[layer_idx, blk_idx, offset] = kt[i]
            self.pool.v_pool[layer_idx, blk_idx, offset] = vt[i]

        if layer_idx == self.pool.num_layers - 1:
            self._seen_tokens = end_pos

        k_blks = self.pool.k_pool[layer_idx, self.block_table]
        v_blks = self.pool.v_pool[layer_idx, self.block_table]
        k_out = k_blks.permute(2, 0, 1, 3).reshape(1, h, self.capacity, d)
        v_out = v_blks.permute(2, 0, 1, 3).reshape(1, h, self.capacity, d)
        return k_out, v_out

    def release(self) -> None:
        if self.block_table:
            self.pool.free(self.block_table)
            self.block_table.clear()
            self._seen_tokens = 0
            self.capacity = 0


class WorkerKVCacheStore:
    """Manages session-keyed KV caches and VRAM budget guard for a pipeline worker stage."""

    def __init__(
        self,
        max_tokens_budget: int = 100_000,
        model_config: Optional[Any] = None,
        device: Optional[str] = None,
        use_static_cache: bool = False,
        num_layers: int = 4,
    ):
        self.max_tokens_budget = max_tokens_budget
        self.model_config = model_config
        self.device = device
        self.use_static_cache = use_static_cache
        self.num_layers = num_layers
        self.allocated_tokens: int = 0
        self.sessions: Dict[str, Any] = {}
        self.session_caps: Dict[str, int] = {}
        self.session_seq_lens: Dict[str, int] = {}
        self.block_pool: Optional[PagedBlockPool] = None

        if HAS_TORCH:
            num_kv_heads = (
                getattr(model_config, "num_key_value_heads", getattr(model_config, "num_attention_heads", 8))
                if model_config is not None
                else 8
            )
            head_dim = (
                getattr(model_config, "head_dim", None)
                or (model_config.hidden_size // model_config.num_attention_heads if model_config is not None and hasattr(model_config, "hidden_size") else 128)
            )
            try:
                self.block_pool = PagedBlockPool(
                    num_layers=num_layers,
                    num_blocks=2048,
                    block_size=16,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    device=device or "cpu",
                    dtype=torch.float16,
                )
            except Exception as e:
                print(f"[WorkerKVCacheStore] Warning: Could not allocate PagedBlockPool: {e}")
                self.block_pool = None

    def get_or_create(self, request_id: str, prompt_len: int, max_tokens: Optional[int] = None) -> Any:
        """Retrieve existing session cache or allocate a new tile-aligned capacity."""
        if request_id in self.sessions:
            return self.sessions[request_id]

        hard_cap = prompt_len + (max_tokens if max_tokens is not None else 512)
        aligned_cap = ((hard_cap + 15) // 16) * 16
        for b in (64, 128, 256, 512, 1024, 2048):
            if b >= hard_cap:
                aligned_cap = b
                break
        else:
            aligned_cap = ((hard_cap + 63) // 64) * 64

        if HAS_TORCH and self.block_pool is not None:
            cache = PagedSessionCache(self.block_pool, initial_capacity=aligned_cap)
        elif HAS_TORCH and self.use_static_cache and self.model_config is not None:
            try:
                from transformers.cache_utils import StaticCache
                cache = StaticCache(
                    config=self.model_config,
                    max_cache_len=aligned_cap,
                    max_batch_size=1,
                    device=self.device or "cpu",
                    dtype=torch.float16,
                )
            except Exception as e:
                print(f"[WorkerKVCacheStore] Warning: Error initializing StaticCache: {e}. Falling back to DynamicCache.")
                from transformers.cache_utils import DynamicCache
                cache = DynamicCache()
        elif HAS_TORCH:
            from transformers.cache_utils import DynamicCache
            cache = DynamicCache()
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
            cache = self.sessions.pop(request_id, None)
            if hasattr(cache, "release"):
                cache.release()
            cap = self.session_caps.pop(request_id, 0)
            self.session_seq_lens.pop(request_id, None)
            self.allocated_tokens = max(0, self.allocated_tokens - cap)
            return True
        return False

    def clear(self) -> None:
        for cache in self.sessions.values():
            if hasattr(cache, "release"):
                cache.release()
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
    tcp_port: Optional[int] = None,
    downstream_tcp: Optional[str] = None,
    quantize_activations: bool = False,
    use_static_cache: Optional[bool] = None,
) -> FastAPI:
    """Creates a FastAPI app and optional persistent binary TCP server for a pipeline stage worker."""
    is_first_stage = (stage_id == 0)
    is_final_stage = (stage_id == total_stages - 1)
    quantize_activations = quantize_activations or (os.environ.get("USE_FP8_ACTIVATIONS", "0") == "1")

    target_downstream_tcp = downstream_tcp or (
        downstream_url.replace("tcp://", "") if downstream_url and downstream_url.startswith("tcp://") else None
    )
    downstream_client = BinaryTransportClient(target_downstream_tcp) if target_downstream_tcp else None

    app = FastAPI(title=f"FrontierSplit-Worker-Stage-{stage_id}")

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


        # Check if sharded safetensors index exists for fast selective stage loading
        use_selective_sharded_loading = False
        weight_map = {}
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            import json
            index_path = hf_hub_download(model_name_or_path, "model.safetensors.index.json")
            with open(index_path, "r", encoding="utf-8") as f:
                weight_map = json.load(f).get("weight_map", {})
            use_selective_sharded_loading = bool(weight_map)
        except Exception as e:
            print(f"[Worker Stage {stage_id}] Note: Selective sharded loader unavailable ({e}), using standard from_pretrained")
            use_selective_sharded_loading = False

        if use_selective_sharded_loading:
            print(f"[Worker Stage {stage_id}] Fast selective sharded loader active. Materializing layers {start_layer}..{end_layer}...")
            import torch.nn as nn
            mtype = getattr(config, "model_type", "").lower()
            if "mixtral" in mtype:
                from transformers.models.mixtral.modeling_mixtral import (
                    MixtralDecoderLayer as DecoderLayer,
                    MixtralRMSNorm as RMSNorm,
                    MixtralRotaryEmbedding as RotaryEmbedding,
                )
            elif "llama" in mtype:
                from transformers.models.llama.modeling_llama import (
                    LlamaDecoderLayer as DecoderLayer,
                    LlamaRMSNorm as RMSNorm,
                    LlamaRotaryEmbedding as RotaryEmbedding,
                )
            elif "qwen2" in mtype:
                from transformers.models.qwen2.modeling_qwen2 import (
                    Qwen2DecoderLayer as DecoderLayer,
                    Qwen2RMSNorm as RMSNorm,
                    Qwen2RotaryEmbedding as RotaryEmbedding,
                )
            else:
                from transformers.models.mistral.modeling_mistral import (
                    MistralDecoderLayer as DecoderLayer,
                    MistralRMSNorm as RMSNorm,
                    MistralRotaryEmbedding as RotaryEmbedding,
                )

            rotary_emb = RotaryEmbedding(config).to(resolved_device)
            if is_first_stage:
                embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=torch.float16)

            stage_layers = []
            for idx in range(start_layer, end_layer + 1):
                layer = DecoderLayer(config, layer_idx=idx).to(dtype=torch.float16)
                layer.self_attn.layer_idx = idx - start_layer
                stage_layers.append(layer)

            if is_final_stage:
                final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(dtype=torch.float16)
                lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.float16)

            # Determine strictly needed shards for this stage
            needed_shards = set()
            for k, shard in weight_map.items():
                if any(k.startswith(f"model.layers.{l}.") for l in range(start_layer, end_layer + 1)):
                    needed_shards.add(shard)
                if is_first_stage and k.startswith("model.embed_tokens."):
                    needed_shards.add(shard)
                if is_final_stage and (k.startswith("model.norm.") or k.startswith("lm_head.")):
                    needed_shards.add(shard)

            print(f"[Worker Stage {stage_id}] Downloading & injecting {len(needed_shards)} shards: {sorted(needed_shards)}")
            raw_layer_tensors: Dict[int, Dict[str, Any]] = {l: {} for l in range(start_layer, end_layer + 1)}
            for shard in sorted(needed_shards):
                t_sh = time.time()
                s_path = hf_hub_download(model_name_or_path, shard)
                sd = load_file(s_path, device="cpu")
                for l_idx in range(start_layer, end_layer + 1):
                    pfx = f"model.layers.{l_idx}."
                    for k, v in sd.items():
                        if k.startswith(pfx):
                            raw_layer_tensors[l_idx][k[len(pfx):]] = v
                if is_first_stage and "model.embed_tokens.weight" in sd:
                    embed_tokens.weight.data.copy_(sd["model.embed_tokens.weight"].to(torch.float16))
                if is_final_stage:
                    if "model.norm.weight" in sd:
                        final_norm.weight.data.copy_(sd["model.norm.weight"].to(torch.float16))
                    if "lm_head.weight" in sd:
                        lm_head.weight.data.copy_(sd["lm_head.weight"].to(torch.float16))
                del sd
                print(f"[Worker Stage {stage_id}] Loaded shard {shard} in {time.time()-t_sh:.1f}s")

            # Convert and inject weights into stage layers with strict validation
            for l_idx in range(start_layer, end_layer + 1):
                raw_sub = raw_layer_tensors[l_idx]
                layer = stage_layers[l_idx - start_layer]
                target_sd = {}

                has_stacked_experts = hasattr(layer, "mlp") and hasattr(layer.mlp, "experts") and hasattr(layer.mlp.experts, "gate_up_proj")
                if has_stacked_experts and any(k.startswith("block_sparse_moe.") for k in raw_sub):
                    # Convert raw checkpoint format (block_sparse_moe.*) to unified stacked MoE (transformers >= 4.49 / 5.x)
                    for k in [
                        "input_layernorm.weight",
                        "post_attention_layernorm.weight",
                        "self_attn.q_proj.weight",
                        "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight",
                        "self_attn.o_proj.weight",
                    ]:
                        if k in raw_sub:
                            target_sd[k] = raw_sub[k].to(torch.float16)

                    if "block_sparse_moe.gate.weight" in raw_sub:
                        target_sd["mlp.gate.weight"] = raw_sub["block_sparse_moe.gate.weight"].to(torch.float16)

                    num_exp = getattr(config, "num_local_experts", 8)
                    gate_up_list = []
                    down_list = []
                    for e in range(num_exp):
                        w1 = raw_sub[f"block_sparse_moe.experts.{e}.w1.weight"].to(torch.float16)
                        w3 = raw_sub[f"block_sparse_moe.experts.{e}.w3.weight"].to(torch.float16)
                        w2 = raw_sub[f"block_sparse_moe.experts.{e}.w2.weight"].to(torch.float16)
                        gate_up_list.append(torch.cat([w1, w3], dim=0))
                        down_list.append(w2)

                    target_sd["mlp.experts.gate_up_proj"] = torch.stack(gate_up_list, dim=0)
                    target_sd["mlp.experts.down_proj"] = torch.stack(down_list, dim=0)
                else:
                    target_sd = {k: v.to(torch.float16) for k, v in raw_sub.items()}

                layer.load_state_dict(target_sd, strict=True)
                raw_layer_tensors[l_idx].clear()

            raw_layer_tensors.clear()

            gc.collect()
            # Move loaded modules to target GPU device
            if is_first_stage and embed_tokens is not None:
                embed_tokens = embed_tokens.to(resolved_device)
            for l in stage_layers:
                assigned_layers.append(l.to(resolved_device))
            if is_final_stage:
                if final_norm is not None:
                    final_norm = final_norm.to(resolved_device)
                if lm_head is not None:
                    lm_head = lm_head.to(resolved_device)

            model_config = config

        else:
            # Standard from_pretrained fallback
            full_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

            model_config = getattr(full_model, "config", None)
            base_model = getattr(full_model, "model", getattr(full_model, "transformer", full_model))
            all_layers = getattr(base_model, "layers", getattr(base_model, "h", []))
            raw_rotary = getattr(base_model, "rotary_emb", None)
            rotary_emb = raw_rotary.to(resolved_device) if raw_rotary is not None else None

            if is_first_stage:
                raw_embed = getattr(base_model, "embed_tokens", getattr(base_model, "wte", None))
                embed_tokens = raw_embed.to(resolved_device) if raw_embed is not None else None

            for idx in range(start_layer, end_layer + 1):
                layer = all_layers[idx].to(resolved_device)
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                    layer.self_attn.layer_idx = idx - start_layer
                assigned_layers.append(layer)

            if is_final_stage:
                raw_norm = getattr(base_model, "norm", getattr(base_model, "ln_f", None))
                final_norm = raw_norm.to(resolved_device) if raw_norm is not None else None
                raw_lm_head = getattr(full_model, "lm_head", None)
                lm_head = raw_lm_head.to(resolved_device) if raw_lm_head is not None else None

            del full_model
            del base_model
            del all_layers
            gc.collect()

        mask_function = None
        try:
            from transformers.models.mistral.modeling_mistral import create_causal_mask, create_sliding_window_causal_mask
            mask_function = create_causal_mask if getattr(model_config, "sliding_window", None) is None else create_sliding_window_causal_mask
        except Exception as e:
            print(f"[Worker Stage {stage_id}] Warning: Could not import Mistral causal mask functions: {e}")

        # Patch MoE experts with static dense-masked routing on TPU / XLA
        import types
        for l in assigned_layers:
            moe_block = getattr(l, "block_sparse_moe", getattr(l, "mlp", None))
            if moe_block is not None and hasattr(moe_block, "experts") and hasattr(moe_block.experts, "gate_up_proj"):
                exp_mod = moe_block.experts
                def _make_static_forward(exp):
                    def _forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
                        num_tokens = hidden_states.shape[0]
                        dense_weights = torch.zeros((num_tokens, exp.num_experts), dtype=hidden_states.dtype, device=hidden_states.device)
                        dense_weights.scatter_(1, top_k_index, top_k_weights.to(hidden_states.dtype))

                        final_hidden_states = torch.zeros_like(hidden_states)
                        for expert_idx in range(exp.num_experts):
                            w = dense_weights[:, expert_idx:expert_idx + 1]
                            gate, up = F.linear(hidden_states, exp.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                            h = exp.act_fn(gate) * up
                            h = F.linear(h, exp.down_proj[expert_idx])
                            final_hidden_states = final_hidden_states + h * w
                        return final_hidden_states
                    return _forward
                exp_mod.forward = types.MethodType(_make_static_forward(exp_mod), exp_mod)

        # Ahead-of-Time JIT Graph Pre-compilation on TPU startup for static Chunk size C=16
        if resolved_device.startswith("xla"):
            print(f"[Worker Stage {stage_id}] Ahead-of-Time JIT pre-compiling XLA static [16, {model_config.hidden_size}] execution graphs...")
            try:
                with torch.no_grad():
                    dummy_chunk = torch.zeros((16, model_config.hidden_size), dtype=torch.float16, device=resolved_device)
                    pos_ids = torch.arange(0, 16, dtype=torch.long, device=resolved_device)
                    pos_emb = rotary_emb(dummy_chunk, position_ids=pos_ids) if rotary_emb is not None else None
                    cur_h = dummy_chunk
                    for l in assigned_layers:
                        out = l(cur_h, position_ids=pos_ids, position_embeddings=pos_emb)
                        cur_h = out[0] if isinstance(out, tuple) else out
                    if is_final_stage:
                        if final_norm is not None:
                            cur_h = final_norm(cur_h)
                        if lm_head is not None:
                            _ = lm_head(cur_h[0:1, :])
                    if is_first_stage and embed_tokens is not None:
                        _ = embed_tokens(torch.zeros(16, device=resolved_device, dtype=torch.long))

                    import torch_xla.core.xla_model as xm
                    xm.mark_step()
                    print(f"[Worker Stage {stage_id}] XLA static [16, {model_config.hidden_size}] execution graphs successfully compiled into TPU HBM!")
            except Exception as e:
                print(f"[Worker Stage {stage_id}] Note: Ahead-of-time pre-compilation note: {e}")

        print(f"[Worker Stage {stage_id}] Successfully mounted assigned layers into VRAM.")

    else:
        # Synthetic fallback mode for testing and development
        np.random.seed(42 + stage_id)
        layer_proj = np.eye(hidden_size, dtype=np.float32) + 0.001 * np.random.randn(hidden_size, hidden_size).astype(np.float32)
        embeddings = np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.02 if is_first_stage else None
        lm_head_synth = np.random.randn(hidden_size, vocab_size).astype(np.float32) * 0.02 if is_final_stage else None

    if use_static_cache is None:
        use_static_cache = bool(resolved_device and resolved_device.startswith("xla"))

    kv_store = WorkerKVCacheStore(
        model_config=config if use_real_model else None,
        device=resolved_device,
        use_static_cache=use_static_cache,
        num_layers=len(assigned_layers) if use_real_model else 4,
    )
    compute_lock = asyncio.Lock()
    reply_clients: Dict[str, BinaryTransportClient] = {}

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

    async def do_release_sessions(packet: ReleaseSessionPacket) -> ReleaseSessionResponse:
        released = 0
        for req_id in packet.request_ids:
            if kv_store.release(req_id):
                released += 1

        if not is_final_stage:
            if downstream_client is not None:
                try:
                    await downstream_client.send_release_sessions(packet.request_ids)
                except Exception as e:
                    print(f"[Worker Stage {stage_id}] Warning: Propagating binary release_sessions failed: {e}")
            elif downstream_url:
                def _post_rel():
                    try:
                        requests.post(f"{downstream_url}/release_sessions", json=packet.model_dump(), timeout=10)
                    except Exception as e:
                        print(f"[Worker Stage {stage_id}] Warning: Propagating release_sessions to {downstream_url} failed: {e}")
                await asyncio.to_thread(_post_rel)

        return ReleaseSessionResponse(
            status="ok",
            released_count=released,
            active_sessions=len(kv_store.sessions),
        )

    @app.post("/release_sessions", response_model=ReleaseSessionResponse)
    async def release_sessions(packet: ReleaseSessionPacket):
        return await do_release_sessions(packet)

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
                        is_static = _is_static_cache(cache)
                        if is_static and hasattr(cache, "capacity"):
                            cap = cache.get_max_length()
                            q_idx = torch.arange(prompt_len, device=resolved_device).view(prompt_len, 1)
                            k_idx = torch.arange(cap, device=resolved_device).view(1, cap)
                            causal_mask = torch.where(k_idx <= q_idx, 0.0, -10000.0).view(1, 1, prompt_len, cap).to(dtype=hidden_states.dtype)
                            cache_pos = None
                        elif is_static:
                            causal_mask = mask_function(
                                config=model_config,
                                inputs_embeds=hidden_states,
                                attention_mask=None,
                                past_key_values=cache,
                                position_ids=pos_ids,
                            ) if mask_function is not None else None
                            cache_pos = torch.arange(prompt_len, dtype=torch.long, device=resolved_device)
                        else:
                            causal_mask = mask_function(
                                config=model_config,
                                inputs_embeds=hidden_states,
                                attention_mask=None,
                                past_key_values=cache,
                                position_ids=pos_ids,
                            ) if mask_function is not None else None
                            cache_pos = None
                        pos_emb = rotary_emb(hidden_states, position_ids=pos_ids) if rotary_emb is not None else None
                        for layer in assigned_layers:
                            kwargs = {
                                "attention_mask": causal_mask,
                                "position_ids": pos_ids,
                                "past_key_values": cache,
                                "use_cache": True,
                                "position_embeddings": pos_emb,
                            }
                            if cache_pos is not None:
                                kwargs["cache_position"] = cache_pos
                            layer_out = layer(hidden_states, **kwargs)
                            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                    else:
                        cache = kv_store.get(packet.request_id)
                        if cache is not None:
                            curr_len = kv_store.get_seq_len(packet.request_id)
                            pos_ids = torch.tensor([[curr_len]], dtype=torch.long, device=resolved_device)
                            is_static = _is_static_cache(cache)
                            if is_static and hasattr(cache, "capacity"):
                                cap = cache.get_max_length()
                                k_idx = torch.arange(cap, device=resolved_device).view(1, cap)
                                causal_mask = torch.where(k_idx <= curr_len, 0.0, -10000.0).view(1, 1, 1, cap).to(dtype=hidden_states.dtype)
                                cache_pos = None
                            elif is_static:
                                causal_mask = mask_function(
                                    config=model_config,
                                    inputs_embeds=hidden_states,
                                    attention_mask=None,
                                    past_key_values=cache,
                                    position_ids=pos_ids,
                                ) if mask_function is not None else None
                                cache_pos = torch.tensor([curr_len], dtype=torch.long, device=resolved_device)
                            else:
                                causal_mask = mask_function(
                                    config=model_config,
                                    inputs_embeds=hidden_states,
                                    attention_mask=None,
                                    past_key_values=cache,
                                    position_ids=pos_ids,
                                ) if mask_function is not None else None
                                cache_pos = None
                            pos_emb = rotary_emb(hidden_states, position_ids=pos_ids) if rotary_emb is not None else None
                            for layer in assigned_layers:
                                kwargs = {
                                    "attention_mask": causal_mask,
                                    "position_ids": pos_ids,
                                    "past_key_values": cache,
                                    "use_cache": True,
                                    "position_embeddings": pos_emb,
                                }
                                if cache_pos is not None:
                                    kwargs["cache_position"] = cache_pos
                                layer_out = layer(hidden_states, **kwargs)
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

    async def do_forward_batched(packet: BatchedActivationPacket) -> BatchedGenerationResponse:
        start_time = time.time()
        batch_size = len(packet.request_ids)

        try:
            if use_real_model:
                async with compute_lock:
                    with torch.no_grad():
                        if is_first_stage and packet.tokens_batch:
                            token_tensor = torch.tensor(packet.tokens_batch, device=resolved_device, dtype=torch.long)
                            hidden_states = embed_tokens(token_tensor)
                        else:
                            raw_bytes = packet.get_raw_bytes()
                            if raw_bytes:
                                if packet.tensor_scale is not None and packet.tensor_dtype in ("int8", "float8_e4m3fn"):
                                    if packet.tensor_dtype == "float8_e4m3fn" and hasattr(torch, "float8_e4m3fn"):
                                        q = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(packet.tensor_shape).to(resolved_device, non_blocking=True)
                                        hidden_states = (q.to(torch.float16) * packet.tensor_scale)
                                    else:
                                        q = torch.frombuffer(bytearray(raw_bytes), dtype=torch.int8).reshape(packet.tensor_shape).to(resolved_device, non_blocking=True)
                                        hidden_states = (q.to(torch.float16) * packet.tensor_scale)
                                else:
                                    dtype_obj = getattr(torch, packet.tensor_dtype, torch.float16)
                                    hidden_states = torch.frombuffer(
                                        bytearray(raw_bytes), dtype=dtype_obj
                                    ).reshape(packet.tensor_shape).to(resolved_device, non_blocking=True)
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

                                # Per-stream attention using each session's contiguous DynamicCache or StaticCache
                                attn_outs = []
                                num_kv_groups = getattr(layer.self_attn, "num_key_value_groups", 1)
                                for b_idx, req_id in enumerate(packet.request_ids):
                                    cache = kv_store.get(req_id)
                                    q_b = q[b_idx:b_idx+1]
                                    k_b = k[b_idx:b_idx+1]
                                    v_b = v[b_idx:b_idx+1]
                                    is_static = _is_static_cache(cache)
                                    if cache is not None:
                                        if is_static and hasattr(cache, "capacity"):
                                            k_cached, v_cached = cache.update(k_b, v_b, layer.self_attn.layer_idx)
                                        elif is_static:
                                            curr_pos = torch.tensor([kv_store.get_seq_len(req_id)], dtype=torch.long, device=resolved_device)
                                            k_cached, v_cached = cache.update(k_b, v_b, layer.self_attn.layer_idx, cache_kwargs={"cache_position": curr_pos})
                                        else:
                                            k_cached, v_cached = cache.update(k_b, v_b, layer.self_attn.layer_idx)
                                    else:
                                        k_cached, v_cached = k_b, v_b

                                    if num_kv_groups > 1:
                                        k_cached = k_cached.repeat_interleave(num_kv_groups, dim=1)
                                        v_cached = v_cached.repeat_interleave(num_kv_groups, dim=1)

                                    if is_static:
                                        curr_len = kv_store.get_seq_len(req_id)
                                        max_k_len = k_cached.shape[2]
                                        col_indices = torch.arange(max_k_len, device=resolved_device).view(1, 1, 1, max_k_len)
                                        mask = torch.where(col_indices > curr_len, -10000.0, 0.0).to(dtype=q_b.dtype)
                                        out_b = F.scaled_dot_product_attention(q_b, k_cached, v_cached, attn_mask=mask)
                                    else:
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
                                    moe_out = layer.mlp(normed)
                                    moe_out = moe_out[0] if isinstance(moe_out, tuple) else moe_out
                                    hidden_states = hidden_states + moe_out
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
                                is_static = _is_static_cache(cache)
                                if is_static and hasattr(cache, "capacity"):
                                    cap = cache.get_max_length()
                                    q_idx = torch.arange(prompt_len, device=resolved_device).view(prompt_len, 1)
                                    k_idx = torch.arange(cap, device=resolved_device).view(1, cap)
                                    c_mask = torch.where(k_idx <= q_idx, 0.0, -10000.0).view(1, 1, prompt_len, cap).to(dtype=h_b.dtype)
                                    cache_pos = None
                                elif is_static:
                                    c_mask = mask_function(
                                        config=model_config,
                                        inputs_embeds=h_b,
                                        attention_mask=None,
                                        past_key_values=cache,
                                        position_ids=pos_ids,
                                    ) if mask_function is not None else None
                                    cache_pos = torch.arange(prompt_len, dtype=torch.long, device=resolved_device)
                                else:
                                    c_mask = mask_function(
                                        config=model_config,
                                        inputs_embeds=h_b,
                                        attention_mask=None,
                                        past_key_values=cache,
                                        position_ids=pos_ids,
                                    ) if mask_function is not None else None
                                    cache_pos = None

                                pos_emb = rotary_emb(h_b, position_ids=pos_ids) if rotary_emb is not None else None

                                for layer in assigned_layers:
                                    kwargs = {
                                        "attention_mask": c_mask,
                                        "position_ids": pos_ids,
                                        "past_key_values": cache,
                                        "use_cache": True,
                                        "position_embeddings": pos_emb,
                                    }
                                    if cache_pos is not None:
                                        kwargs["cache_position"] = cache_pos
                                    l_out = layer(h_b, **kwargs)
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

                        if resolved_device.startswith("xla"):
                            try:
                                import torch_xla.core.xla_model as xm
                                xm.mark_step()
                            except ImportError:
                                pass

                        if is_final_stage:
                            if final_norm is not None:
                                hidden_states = final_norm(hidden_states)

                            last_token_hidden = hidden_states[:, -1, :]  # shape: [B, hidden_size]
                            logits = lm_head(last_token_hidden)  # shape: [B, vocab_size]

                            next_token_ids = torch.argmax(logits, dim=-1).tolist()

                stage_compute_ms = (time.time() - start_time) * 1000
                timings = dict(packet.stage_timings)
                timings[f"stage_{stage_id}_compute_ms"] = stage_compute_ms

                if not is_final_stage:
                    next_packet = BatchedActivationPacket(
                        request_ids=packet.request_ids,
                        sequence_steps=packet.sequence_steps,
                        stage_id=stage_id + 1,
                        is_prefill=packet.is_prefill,
                        use_kv_cache=packet.use_kv_cache,
                        max_tokens_list=packet.max_tokens_list,
                        attention_mask=packet.attention_mask,
                        stage_timings=timings,
                        reply_to=packet.reply_to,
                    )
                    if quantize_activations:
                        amax = hidden_states.abs().max().item()
                        scale = 1.0 if amax == 0 else amax / 127.0
                        q = torch.clamp(torch.round(hidden_states / scale), -128, 127).to(torch.int8)
                        raw_bytes = q.contiguous().cpu().numpy().tobytes()
                        next_packet.tensor_scale = scale
                        next_packet.set_raw_tensor(raw_bytes, list(hidden_states.shape), "int8")
                    else:
                        cpu_t = hidden_states.contiguous().cpu()
                        dtype_str = str(hidden_states.dtype).replace("torch.", "")
                        if dtype_str in ("float16", "bfloat16"):
                            raw_bytes = cpu_t.view(torch.uint8).numpy().tobytes()
                        else:
                            raw_bytes = cpu_t.numpy().tobytes()
                        next_packet.set_raw_tensor(raw_bytes, list(hidden_states.shape), dtype_str)

                    if downstream_client is not None:
                        if packet.reply_to:
                            await downstream_client.send_async_forward(next_packet, raw_tensor_bytes=raw_bytes)
                            return BatchedGenerationResponse(responses=[], batch_size=0, stage_timings=timings)
                        else:
                            return await downstream_client.send_batched_forward(next_packet)
                    elif downstream_url:
                        def _post():
                            resp = requests.post(f"{downstream_url}/forward_batched", json=next_packet.model_dump(), timeout=600)
                            resp.raise_for_status()
                            return resp.json()
                        res_json = await asyncio.to_thread(_post)
                        return BatchedGenerationResponse.model_validate(res_json)
                    else:
                        raise HTTPException(status_code=500, detail="Missing downstream_url for intermediate stage")

                else:
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

                    batch_resp = BatchedGenerationResponse(
                        responses=responses,
                        batch_size=len(responses),
                        stage_timings=timings,
                    )

                    if packet.reply_to:
                        if packet.reply_to not in reply_clients:
                            reply_clients[packet.reply_to] = BinaryTransportClient(packet.reply_to)
                        reply_client = reply_clients[packet.reply_to]
                        await reply_client.send_response_frame(batch_resp)

                    return batch_resp

            else:
                # Synthetic batched matrix compute
                async with compute_lock:
                    if is_first_stage and packet.tokens_batch:
                        max_tok_len = max(len(t) for t in packet.tokens_batch)
                        padded_batch = [t + [0] * (max_tok_len - len(t)) for t in packet.tokens_batch]
                        tokens_arr = np.array(padded_batch)
                        activation = embeddings[tokens_arr]
                    else:
                        raw_bytes = packet.get_raw_bytes()
                        if raw_bytes:
                            if packet.tensor_scale is not None and packet.tensor_dtype in ("int8", "float8_e4m3fn"):
                                q_arr = np.frombuffer(raw_bytes, dtype=np.int8).reshape(packet.tensor_shape)
                                activation = (q_arr.astype(np.float32) * packet.tensor_scale).astype(np.float32)
                            else:
                                activation = np.frombuffer(raw_bytes, dtype=packet.tensor_dtype).reshape(packet.tensor_shape)
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

                    if is_final_stage:
                        last_token_hidden = activation[:, -1, :]
                        logits = np.matmul(last_token_hidden, lm_head_synth)
                        next_token_ids = np.argmax(logits, axis=-1).tolist()

                if not is_final_stage:
                    next_packet = BatchedActivationPacket(
                        request_ids=packet.request_ids,
                        sequence_steps=packet.sequence_steps,
                        stage_id=stage_id + 1,
                        is_prefill=packet.is_prefill,
                        use_kv_cache=packet.use_kv_cache,
                        max_tokens_list=packet.max_tokens_list,
                        attention_mask=packet.attention_mask,
                        stage_timings=timings,
                        reply_to=packet.reply_to,
                    )
                    if quantize_activations:
                        amax = float(np.max(np.abs(activation)))
                        scale = 1.0 if amax == 0 else amax / 127.0
                        q = np.clip(np.round(activation / scale), -128, 127).astype(np.int8)
                        raw_bytes = q.tobytes()
                        next_packet.tensor_scale = scale
                        next_packet.set_raw_tensor(raw_bytes, list(activation.shape), "int8")
                    else:
                        raw_bytes = activation.tobytes()
                        next_packet.set_raw_tensor(raw_bytes, list(activation.shape), str(activation.dtype))

                    if downstream_client is not None:
                        if packet.reply_to:
                            await downstream_client.send_async_forward(next_packet, raw_tensor_bytes=raw_bytes)
                            return BatchedGenerationResponse(responses=[], batch_size=0, stage_timings=timings)
                        else:
                            return await downstream_client.send_batched_forward(next_packet)
                    elif downstream_url:
                        def _post_synth():
                            resp = requests.post(f"{downstream_url}/forward_batched", json=next_packet.model_dump(), timeout=30)
                            resp.raise_for_status()
                            return resp.json()
                        res_json = await asyncio.to_thread(_post_synth)
                        return BatchedGenerationResponse.model_validate(res_json)
                    else:
                        raise HTTPException(status_code=500, detail="Missing downstream_url for intermediate stage")

                else:
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
                    batch_resp = BatchedGenerationResponse(
                        responses=responses,
                        batch_size=len(responses),
                        stage_timings=timings,
                    )

                    if packet.reply_to:
                        if packet.reply_to not in reply_clients:
                            reply_clients[packet.reply_to] = BinaryTransportClient(packet.reply_to)
                        reply_client = reply_clients[packet.reply_to]
                        await reply_client.send_response_frame(batch_resp)

                    return batch_resp

        except Exception as e:
            if packet.reply_to:
                err_resp = BatchedGenerationResponse(
                    responses=[
                        GenerationResponse(
                            request_id=r_id,
                            token_id=0,
                            text=f"[Error in Stage {stage_id}: {e}]",
                            is_finished=True,
                            latency_ms=0.0,
                        )
                        for r_id in packet.request_ids
                    ],
                    batch_size=len(packet.request_ids),
                )
                try:
                    if packet.reply_to not in reply_clients:
                        reply_clients[packet.reply_to] = BinaryTransportClient(packet.reply_to)
                    await reply_clients[packet.reply_to].send_response_frame(err_resp)
                except Exception:
                    pass
            raise

    async def do_forward_chunk(packet: ChunkActivationPacket) -> ChunkGenerationResponse:
        """Executes a single atomic chunk of fixed size C (default 16 tokens) through assigned pipeline stages.
        Guarantees strictly invariant [C, hidden_size] tensor shape everywhere with ZERO variable batch dimension B.
        """
        start_time = time.time()
        chunk_size = packet.chunk_size
        valid_tokens = packet.valid_tokens
        chunk_idx = packet.chunk_idx
        total_chunks = packet.total_chunks
        is_prefill = packet.is_prefill
        seq_pos = getattr(packet, "seq_pos", chunk_idx * chunk_size)
        is_final_chunk = (chunk_idx == total_chunks - 1) if is_prefill else True

        try:
            if use_real_model:
                async with compute_lock:
                    with torch.no_grad():
                        if is_first_stage and packet.tokens:
                            tok_t = torch.tensor(packet.tokens, device=resolved_device, dtype=torch.long)
                            hidden_states = embed_tokens(tok_t)
                        else:
                            raw_bytes = packet.get_raw_bytes()
                            if raw_bytes:
                                dtype_obj = getattr(torch, packet.tensor_dtype, torch.float16)
                                hidden_states = torch.frombuffer(
                                    bytearray(raw_bytes), dtype=dtype_obj
                                ).reshape(packet.tensor_shape).to(resolved_device, non_blocking=True)
                            else:
                                arr = packet.get_tensor()
                                hidden_states = torch.from_numpy(arr).to(device=resolved_device, dtype=torch.float16)

                        # Enforce invariant 2D shape [C, hidden_size]
                        hidden_states = hidden_states.view(chunk_size, -1)

                        # Look up session cache in PagedBlockPool
                        cache = kv_store.get_or_create(packet.request_id, prompt_len=max(total_chunks * chunk_size, seq_pos + 64))

                        start_pos = chunk_idx * chunk_size if is_prefill else seq_pos
                        pos_ids = torch.arange(start_pos, start_pos + chunk_size, device=resolved_device, dtype=torch.long).unsqueeze(0)
                        cos, sin = rotary_emb(hidden_states.unsqueeze(0), pos_ids) if rotary_emb is not None else (None, None)

                        for local_layer_idx, layer in enumerate(assigned_layers):
                            residual = hidden_states
                            hidden_norm = layer.input_layernorm(hidden_states)

                            num_q_heads = layer.self_attn.config.num_attention_heads
                            num_kv_heads = layer.self_attn.config.num_key_value_heads
                            head_dim = layer.self_attn.head_dim

                            # Batched Q, K, V projections for Chunk [C, 4096]
                            q = layer.self_attn.q_proj(hidden_norm).view(chunk_size, num_q_heads, head_dim).transpose(0, 1)
                            k = layer.self_attn.k_proj(hidden_norm).view(chunk_size, num_kv_heads, head_dim).transpose(0, 1)
                            v = layer.self_attn.v_proj(hidden_norm).view(chunk_size, num_kv_heads, head_dim).transpose(0, 1)

                            if cos is not None and sin is not None:
                                q, k = apply_rotary_pos_emb(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
                                q = q.squeeze(0)
                                k = k.squeeze(0)

                            if hasattr(cache, "pool") and cache.pool is not None:
                                if is_prefill:
                                    if len(cache.block_table) <= chunk_idx:
                                        needed = chunk_idx + 1 - len(cache.block_table)
                                        cache.block_table.extend(cache.pool.allocate(needed))
                                    blk_id = cache.block_table[chunk_idx]
                                    cache.pool.k_pool[local_layer_idx, blk_id, :] = k.permute(1, 0, 2)
                                    cache.pool.v_pool[local_layer_idx, blk_id, :] = v.permute(1, 0, 2)

                                    active_blks = cache.block_table[:chunk_idx + 1]
                                    k_blks = cache.pool.k_pool[local_layer_idx, active_blks]
                                    v_blks = cache.pool.v_pool[local_layer_idx, active_blks]

                                    total_k = (chunk_idx + 1) * chunk_size
                                    k_all = k_blks.reshape(total_k, num_kv_heads, head_dim).permute(1, 0, 2)
                                    v_all = v_blks.reshape(total_k, num_kv_heads, head_dim).permute(1, 0, 2)

                                    num_kv_groups = num_q_heads // num_kv_heads
                                    if num_kv_groups > 1:
                                        k_all = k_all.repeat_interleave(num_kv_groups, dim=0)
                                        v_all = v_all.repeat_interleave(num_kv_groups, dim=0)

                                    q_idx = torch.arange(start_pos, start_pos + chunk_size, device=resolved_device).view(chunk_size, 1)
                                    k_idx = torch.arange(total_k, device=resolved_device).view(1, total_k)
                                    valid_limit = start_pos + valid_tokens - 1
                                    mask = torch.where((k_idx <= q_idx) & (k_idx <= valid_limit), 0.0, -10000.0).view(1, 1, chunk_size, total_k).to(dtype=q.dtype)

                                    attn_out = F.scaled_dot_product_attention(q.unsqueeze(0), k_all.unsqueeze(0), v_all.unsqueeze(0), attn_mask=mask)
                                    attn_out = attn_out.squeeze(0).transpose(0, 1).reshape(chunk_size, -1)
                                else:
                                    blk_idx = seq_pos // chunk_size
                                    slot_idx = seq_pos % chunk_size
                                    if len(cache.block_table) <= blk_idx:
                                        needed = blk_idx + 1 - len(cache.block_table)
                                        cache.block_table.extend(cache.pool.allocate(needed))
                                    blk_id = cache.block_table[blk_idx]
                                    cache.pool.k_pool[local_layer_idx, blk_id, slot_idx] = k[:, 0, :]
                                    cache.pool.v_pool[local_layer_idx, blk_id, slot_idx] = v[:, 0, :]

                                    total_tokens = seq_pos + 1
                                    active_blks = cache.block_table[:blk_idx + 1]
                                    k_blks = cache.pool.k_pool[local_layer_idx, active_blks].reshape(len(active_blks) * chunk_size, num_kv_heads, head_dim)
                                    v_blks = cache.pool.v_pool[local_layer_idx, active_blks].reshape(len(active_blks) * chunk_size, num_kv_heads, head_dim)
                                    k_all = k_blks[:total_tokens].permute(1, 0, 2)
                                    v_all = v_blks[:total_tokens].permute(1, 0, 2)

                                    num_kv_groups = num_q_heads // num_kv_heads
                                    if num_kv_groups > 1:
                                        k_all = k_all.repeat_interleave(num_kv_groups, dim=0)
                                        v_all = v_all.repeat_interleave(num_kv_groups, dim=0)

                                    q_single = q[:, 0:1, :]
                                    attn_single = F.scaled_dot_product_attention(q_single.unsqueeze(0), k_all.unsqueeze(0), v_all.unsqueeze(0))
                                    attn_single = attn_single.squeeze(0).transpose(0, 1).reshape(1, -1)

                                    attn_out = torch.zeros((chunk_size, hidden_states.shape[-1]), dtype=q.dtype, device=resolved_device)
                                    attn_out[0:1, :] = attn_single
                            else:
                                num_kv_groups = num_q_heads // num_kv_heads
                                k_fb = k.repeat_interleave(num_kv_groups, dim=0) if num_kv_groups > 1 else k
                                v_fb = v.repeat_interleave(num_kv_groups, dim=0) if num_kv_groups > 1 else v
                                attn_out = F.scaled_dot_product_attention(q.unsqueeze(0), k_fb.unsqueeze(0), v_fb.unsqueeze(0))
                                attn_out = attn_out.squeeze(0).transpose(0, 1).reshape(chunk_size, -1)

                            hidden_states = residual + layer.self_attn.o_proj(attn_out)

                            normed = layer.post_attention_layernorm(hidden_states)
                            moe_block = getattr(layer, "block_sparse_moe", getattr(layer, "mlp", None))
                            if moe_block is not None:
                                moe_input = normed.unsqueeze(0)
                                moe_out = moe_block(moe_input)
                                moe_out = moe_out[0] if isinstance(moe_out, tuple) else moe_out
                                hidden_states = hidden_states + moe_out.squeeze(0)

                        if resolved_device.startswith("xla"):
                            try:
                                import torch_xla.core.xla_model as xm
                                xm.mark_step()
                            except ImportError:
                                pass

                        next_tok_id = None
                        if is_final_stage:
                            if final_norm is not None:
                                hidden_states = final_norm(hidden_states)
                            if is_final_chunk and lm_head is not None:
                                last_tok_idx = (valid_tokens - 1) if is_prefill else 0
                                logits = lm_head(hidden_states[last_tok_idx:last_tok_idx + 1, :])
                                next_tok_id = int(torch.argmax(logits, dim=-1).item())

                stage_compute_ms = (time.time() - start_time) * 1000
                timings = dict(packet.stage_timings)
                timings[f"stage_{stage_id}_compute_ms"] = stage_compute_ms

                if not is_final_stage:
                    next_packet = packet.model_copy()
                    next_packet.stage_id = stage_id + 1
                    next_packet.seq_pos = seq_pos
                    next_packet.tokens = None
                    next_packet.stage_timings = timings
                    next_packet.tensor_shape = [chunk_size, hidden_states.shape[-1]]

                    raw_bytes = hidden_states.contiguous().cpu().view(torch.uint8).numpy().tobytes()
                    if downstream_client is not None:
                        await downstream_client.send_async_forward(next_packet, raw_tensor_bytes=raw_bytes)
                        return ChunkGenerationResponse(
                            request_id=packet.request_id,
                            chunk_idx=chunk_idx,
                            is_final_chunk=is_final_chunk,
                            stage_timings=timings,
                        )
                    else:
                        resp = requests.post(f"{downstream_url}/forward_chunk", json=next_packet.model_dump(), timeout=600)
                        return ChunkGenerationResponse.model_validate(resp.json())

                else:
                    is_finished = False
                    finish_reason = None
                    if next_tok_id is not None:
                        eos_id = getattr(model_config, "eos_token_id", None)
                        if eos_id is not None and next_tok_id == eos_id:
                            is_finished = True
                            finish_reason = "stop"

                    chunk_resp = ChunkGenerationResponse(
                        request_id=packet.request_id,
                        chunk_idx=chunk_idx,
                        is_final_chunk=is_final_chunk,
                        next_token_id=next_tok_id,
                        is_finished=is_finished,
                        finish_reason=finish_reason,
                        stage_timings=timings,
                    )
                    if packet.reply_to:
                        if packet.reply_to not in reply_clients:
                            reply_clients[packet.reply_to] = BinaryTransportClient(packet.reply_to)
                        await reply_clients[packet.reply_to].send_response_frame(chunk_resp)
                    return chunk_resp

            else:
                # Synthetic mode for unit tests
                stage_compute_ms = (time.time() - start_time) * 1000
                timings = dict(packet.stage_timings)
                timings[f"stage_{stage_id}_compute_ms"] = stage_compute_ms
                next_tok_id = 42 if is_final_chunk else None
                chunk_resp = ChunkGenerationResponse(
                    request_id=packet.request_id,
                    chunk_idx=chunk_idx,
                    is_final_chunk=is_final_chunk,
                    next_token_id=next_tok_id,
                    stage_timings=timings,
                )
                if not is_final_stage:
                    next_packet = packet.model_copy()
                    next_packet.stage_id = stage_id + 1
                    next_packet.stage_timings = timings
                    if downstream_client is not None:
                        await downstream_client.send_async_forward(next_packet)
                    return chunk_resp
                else:
                    if packet.reply_to:
                        if packet.reply_to not in reply_clients:
                            reply_clients[packet.reply_to] = BinaryTransportClient(packet.reply_to)
                        await reply_clients[packet.reply_to].send_response_frame(chunk_resp)
                    return chunk_resp

        except Exception as e:
            traceback.print_exc()
            if packet.reply_to:
                err_resp = ChunkGenerationResponse(
                    request_id=packet.request_id,
                    chunk_idx=chunk_idx,
                    is_final_chunk=is_final_chunk,
                    is_finished=True,
                    finish_reason=f"error: {e}",
                )
                try:
                    if packet.reply_to not in reply_clients:
                        reply_clients[packet.reply_to] = BinaryTransportClient(packet.reply_to)
                    await reply_clients[packet.reply_to].send_response_frame(err_resp)
                except Exception:
                    pass
            raise

    async def unified_forward_handler(packet: Any):
        if isinstance(packet, ChunkActivationPacket):
            return await do_forward_chunk(packet)
        return await do_forward_batched(packet)

    @app.post("/forward_batched", response_model=BatchedGenerationResponse)
    async def forward_batched(packet: BatchedActivationPacket):
        return await do_forward_batched(packet)

    @app.post("/forward_chunk", response_model=ChunkGenerationResponse)
    async def forward_chunk(packet: ChunkActivationPacket):
        return await do_forward_chunk(packet)

    binary_server = None
    if tcp_port is not None:
        binary_server = BinaryTransportServer(
            host="0.0.0.0",
            port=tcp_port,
            forward_handler=unified_forward_handler,
            release_handler=do_release_sessions,
        )

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        if binary_server is not None:
            await binary_server.start()
        yield
        if binary_server is not None:
            await binary_server.stop()
        if downstream_client is not None:
            await downstream_client.close()
        for client in reply_clients.values():
            await client.close()
        reply_clients.clear()

    app.router.lifespan_context = lifespan
    app.state.binary_server = binary_server
    app.state.downstream_client = downstream_client

    return app


def main():
    parser = argparse.ArgumentParser(description="FrontierSplit Pipeline Stage Worker")
    parser.add_argument("--stage-id", type=int, required=True, help="Stage index (0 to total_stages - 1)")
    parser.add_argument("--total-stages", type=int, default=2, help="Total pipeline stages")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on")
    parser.add_argument("--tcp-port", type=int, default=None, help="Persistent binary TCP port to listen on")
    parser.add_argument("--downstream-url", type=str, default=None, help="Downstream worker URL")
    parser.add_argument("--downstream-tcp", type=str, default=None, help="Downstream persistent binary TCP peer (host:port)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    parser.add_argument("--model-name", type=str, default=None, help="Hugging Face model ID to load")
    parser.add_argument("--device", type=str, default=None, help="PyTorch device (cuda, mps, cpu)")
    parser.add_argument("--quantize-activations", action="store_true", default=False, help="Enable dynamic INT8 activation quantization for inter-stage transport")
    parser.add_argument("--use-static-cache", action="store_true", default=None, help="Enable fixed-size StaticCache for fast TPU/XLA compilation elimination")
    parser.add_argument("--disable-static-cache", dest="use_static_cache", action="store_false", help="Disable StaticCache and force DynamicCache")
    args = parser.parse_args()

    app = create_worker_app(
        stage_id=args.stage_id,
        total_stages=args.total_stages,
        downstream_url=args.downstream_url,
        model_name_or_path=args.model_name,
        device=args.device,
        tcp_port=args.tcp_port,
        downstream_tcp=args.downstream_tcp,
        quantize_activations=args.quantize_activations,
        use_static_cache=args.use_static_cache,
    )
    print(f"Starting FrontierSplit Worker Stage {args.stage_id}/{args.total_stages} on port {args.port}...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
