"""Model Slicer and Memory Planner for FrontierSplit.

Inspects model architectures (via Hugging Face config.json), calculates the exact
matrix dimensions and memory requirements for each layer, and generates optimal
pipeline partitioning plans across commodity cluster nodes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LayerMemoryProfile:
    """Calculated memory footprint for a single transformer layer."""
    layer_index: int
    attention_params: int
    moe_ffn_params: int
    shared_ffn_params: int
    dense_ffn_params: int
    router_params: int
    total_params: int
    total_bytes: int
    kv_cache_bytes_per_token: int


@dataclass
class ModelMemoryProfile:
    """Complete architectural memory profile of a model."""
    model_id: str
    architecture: str
    precision: str
    bytes_per_param: float
    total_layers: int
    hidden_size: int
    vocab_size: int
    num_experts: int
    top_k: int
    embedding_bytes: int
    lm_head_bytes: int
    per_layer_bytes: int
    total_model_bytes: int
    activation_bytes_per_token: int
    layers: List[LayerMemoryProfile]


@dataclass
class NodePartition:
    """A slice of the pipeline assigned to a specific cluster node."""
    node_id: int
    start_layer: int
    end_layer: int
    layer_count: int
    holds_embeddings: bool
    holds_lm_head: bool
    weights_bytes: int
    max_kv_cache_tokens_at_capacity: int
    vram_headroom_bytes: int


@dataclass
class ClusterPartitionPlan:
    """The full partitioning plan for a cluster of nodes."""
    model_profile: ModelMemoryProfile
    num_nodes: int
    vram_per_node_gb: float
    nodes: List[NodePartition]
    is_feasible: bool
    total_cluster_vram_gb: float
    total_weights_gb: float


class ModelSlicer:
    """Analyzes model architecture configs and generates cluster partitions."""

    PRECISION_MAP = {
        "fp32": 4.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "fp8": 1.0,
        "int8": 1.0,
        "int4": 0.5,
    }

    @staticmethod
    def fetch_config(model_id_or_path: str) -> Dict[str, Any]:
        """Fetch config.json from local disk or Hugging Face Hub."""
        if model_id_or_path.endswith(".json"):
            with open(model_id_or_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Attempt to load from HF raw URL
        url = f"https://huggingface.co/{model_id_or_path}/raw/main/config.json"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "FrontierSplit-Slicer/0.1"}
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch config for '{model_id_or_path}' from {url}: {e}"
            ) from e

    @classmethod
    def profile_model(
        cls,
        config: Dict[str, Any],
        model_id: str = "custom",
        precision: str = "fp16",
    ) -> ModelMemoryProfile:
        """Parse architectural hyperparameters and calculate exact memory footprint."""
        bytes_per_param = cls.PRECISION_MAP.get(precision.lower(), 2.0)

        # Standard hyperparameter fallbacks across HF architectures
        num_layers = config.get("num_hidden_layers") or config.get("n_layers") or config.get("num_layers", 32)
        hidden_size = config.get("hidden_size") or config.get("d_model", 4096)
        vocab_size = config.get("vocab_size", 32000)

        num_attn_heads = config.get("num_attention_heads") or config.get("n_heads", 32)
        num_kv_heads = config.get("num_key_value_heads") or config.get("n_kv_heads", num_attn_heads)
        head_dim = config.get("head_dim") or (hidden_size // num_attn_heads)

        # MoE parameters
        num_experts = (
            config.get("num_local_experts")
            or config.get("n_routed_experts")
            or config.get("num_experts")
            or 0
        )
        top_k = (
            config.get("num_experts_per_tok")
            or config.get("num_experts_per_token")
            or config.get("top_k", 2 if num_experts > 0 else 0)
        )
        intermediate_size = config.get("intermediate_size") or (hidden_size * 4)

        # DeepSeek / Qwen shared experts
        n_shared_experts = config.get("n_shared_experts", 0)
        shared_expert_intermediate = config.get(
            "shared_expert_intermediate_size",
            intermediate_size * n_shared_experts if n_shared_experts > 0 else 0
        )
        moe_intermediate_size = config.get("moe_intermediate_size", intermediate_size)

        # 1. Embeddings & LM Head
        embedding_params = vocab_size * hidden_size
        lm_head_params = vocab_size * hidden_size if not config.get("tie_word_embeddings", False) else 0

        # 2. Attention block per layer
        # W_q: hidden_size x (num_attn_heads * head_dim)
        # W_k: hidden_size x (num_kv_heads * head_dim)
        # W_v: hidden_size x (num_kv_heads * head_dim)
        # W_o: (num_attn_heads * head_dim) x hidden_size
        w_q = hidden_size * (num_attn_heads * head_dim)
        w_k = hidden_size * (num_kv_heads * head_dim)
        w_v = hidden_size * (num_kv_heads * head_dim)
        w_o = (num_attn_heads * head_dim) * hidden_size
        attn_params = w_q + w_k + w_v + w_o

        # KV cache per token per layer: 2 * (K and V) * num_kv_heads * head_dim * bytes_per_param
        kv_cache_bytes_per_token_layer = int(2 * num_kv_heads * head_dim * bytes_per_param)

        # 3. FFN / MoE block per layer
        # SwiGLU requires 3 matrices: Gate, Up, Down
        # Gate: hidden_size x intermediate
        # Up: hidden_size x intermediate
        # Down: intermediate x hidden_size
        single_expert_params = 3 * (hidden_size * moe_intermediate_size)

        if num_experts > 0:
            router_params = hidden_size * num_experts
            moe_ffn_params = num_experts * single_expert_params
            dense_ffn_params = 0
        else:
            router_params = 0
            moe_ffn_params = 0
            dense_ffn_params = 3 * (hidden_size * intermediate_size)

        shared_ffn_params = 3 * (hidden_size * shared_expert_intermediate) if n_shared_experts > 0 else 0

        per_layer_params = attn_params + router_params + moe_ffn_params + shared_ffn_params + dense_ffn_params
        per_layer_bytes = int(per_layer_params * bytes_per_param)

        layers_profile = []
        for i in range(num_layers):
            layers_profile.append(
                LayerMemoryProfile(
                    layer_index=i,
                    attention_params=attn_params,
                    moe_ffn_params=moe_ffn_params,
                    shared_ffn_params=shared_ffn_params,
                    dense_ffn_params=dense_ffn_params,
                    router_params=router_params,
                    total_params=per_layer_params,
                    total_bytes=per_layer_bytes,
                    kv_cache_bytes_per_token=kv_cache_bytes_per_token_layer,
                )
            )

        embedding_bytes = int(embedding_params * bytes_per_param)
        lm_head_bytes = int(lm_head_params * bytes_per_param)
        total_model_bytes = embedding_bytes + lm_head_bytes + (per_layer_bytes * num_layers)

        # Activation vector passed over network between layers:
        # A sequence of tokens with dimension `hidden_size`
        activation_bytes_per_token = int(hidden_size * bytes_per_param)

        return ModelMemoryProfile(
            model_id=model_id,
            architecture=config.get("architectures", ["Transformer"])[0] if isinstance(config.get("architectures"), list) else "Transformer",
            precision=precision,
            bytes_per_param=bytes_per_param,
            total_layers=num_layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_experts=num_experts,
            top_k=top_k,
            embedding_bytes=embedding_bytes,
            lm_head_bytes=lm_head_bytes,
            per_layer_bytes=per_layer_bytes,
            total_model_bytes=total_model_bytes,
            activation_bytes_per_token=activation_bytes_per_token,
            layers=layers_profile,
        )

    @classmethod
    def plan_cluster_partition(
        cls,
        model_profile: ModelMemoryProfile,
        num_nodes: int,
        vram_per_node_gb: float,
        vram_safety_margin: float = 0.85,
    ) -> ClusterPartitionPlan:
        """Calculates an optimal pipeline layer split across nodes."""
        usable_vram_per_node = int(vram_per_node_gb * (1024 ** 3) * vram_safety_margin)
        total_cluster_vram = num_nodes * vram_per_node_gb

        # Evenly divide layers across nodes
        layers_per_node = model_profile.total_layers // num_nodes
        remainder = model_profile.total_layers % num_nodes

        nodes: List[NodePartition] = []
        current_layer = 0
        is_feasible = True

        for i in range(num_nodes):
            assigned_count = layers_per_node + (1 if i < remainder else 0)
            start_layer = current_layer
            end_layer = current_layer + assigned_count - 1
            current_layer += assigned_count

            holds_embeddings = (i == 0)
            holds_lm_head = (i == num_nodes - 1)

            weights_bytes = assigned_count * model_profile.per_layer_bytes
            if holds_embeddings:
                weights_bytes += model_profile.embedding_bytes
            if holds_lm_head:
                weights_bytes += model_profile.lm_head_bytes

            headroom = usable_vram_per_node - weights_bytes
            if headroom < 0:
                is_feasible = False

            # Max KV tokens supported by remaining VRAM
            kv_bytes_per_token_node = assigned_count * model_profile.layers[0].kv_cache_bytes_per_token
            max_kv_tokens = max(0, headroom // kv_bytes_per_token_node) if kv_bytes_per_token_node > 0 else 0

            nodes.append(
                NodePartition(
                    node_id=i,
                    start_layer=start_layer,
                    end_layer=end_layer,
                    layer_count=assigned_count,
                    holds_embeddings=holds_embeddings,
                    holds_lm_head=holds_lm_head,
                    weights_bytes=weights_bytes,
                    max_kv_cache_tokens_at_capacity=int(max_kv_tokens),
                    vram_headroom_bytes=headroom,
                )
            )

        return ClusterPartitionPlan(
            model_profile=model_profile,
            num_nodes=num_nodes,
            vram_per_node_gb=vram_per_node_gb,
            nodes=nodes,
            is_feasible=is_feasible,
            total_cluster_vram_gb=total_cluster_vram,
            total_weights_gb=model_profile.total_model_bytes / (1024 ** 3),
        )

    @staticmethod
    def print_plan(plan: ClusterPartitionPlan) -> None:
        """Pretty-prints a partitioning plan report."""
        p = plan.model_profile
        print("=" * 70)
        print(f" FRONTIERSPLIT PARTITION PLAN: {p.model_id}")
        print("=" * 70)
        print(f"Architecture:      {p.architecture} ({p.precision.upper()})")
        print(f"Total Layers:      {p.total_layers} layers")
        print(f"Hidden Dim:        {p.hidden_size} numbers")
        print(f"MoE Experts:       {p.num_experts} experts (Top-{p.top_k} active)")
        print(f"Total Model Size:  {plan.total_weights_gb:.2f} GB")
        print(f"Cluster Hardware:  {plan.num_nodes}x Nodes ({plan.vram_per_node_gb:.1f} GB VRAM each = {plan.total_cluster_vram_gb:.1f} GB Total)")
        print(f"VPC Activation:    {p.activation_bytes_per_token / 1024:.2f} KB / token handoff")
        print(f"Feasibility:       {'FEASIBLE' if plan.is_feasible else 'EXCEEDS VRAM (Needs more nodes)'}")
        print("-" * 70)
        print(f"{'Node':<6} {'Layers':<12} {'Weights (GB)':<15} {'VRAM Left (GB)':<16} {'Roles'}")
        print("-" * 70)
        for n in plan.nodes:
            weights_gb = n.weights_bytes / (1024 ** 3)
            headroom_gb = n.vram_headroom_bytes / (1024 ** 3)
            roles = []
            if n.holds_embeddings:
                roles.append("Embeddings")
            if n.holds_lm_head:
                roles.append("LM Head")
            role_str = " + ".join(roles) if roles else "Pipeline Compute"
            print(f"Node {n.node_id:<2} L{n.start_layer:02d}-L{n.end_layer:02d}     {weights_gb:>6.2f} GB        {headroom_gb:>6.2f} GB         {role_str}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="FrontierSplit Model Slicer & Memory Profiler")
    parser.add_argument("model_id", nargs="?", default="mistralai/Mixtral-8x7B-v0.1", help="Hugging Face model ID or path to config.json")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp8", "int8", "int4"], default="fp16", help="Numeric precision")
    parser.add_argument("--nodes", type=int, default=4, help="Number of nodes in pipeline")
    parser.add_argument("--vram", type=float, default=24.0, help="VRAM per node in GB (default: 24.0 for NVIDIA L4)")
    args = parser.parse_args()

    print(f"Fetching config for: {args.model_id}...")
    config = ModelSlicer.fetch_config(args.model_id)
    profile = ModelSlicer.profile_model(config, model_id=args.model_id, precision=args.precision)
    plan = ModelSlicer.plan_cluster_partition(profile, num_nodes=args.nodes, vram_per_node_gb=args.vram)
    ModelSlicer.print_plan(plan)


if __name__ == "__main__":
    main()
