#!/usr/bin/env python3
"""Hardware Optimization & Performance Planner for Split LLM Inference.

Calculates the optimal layer partitioning, memory footprint, compute/memory bottlenecks,
network handoff overhead, and projected Tokens Per Second (TPS) across:
- NVIDIA T4
- NVIDIA L4
- Google TPU v4
- Google TPU v5p
- NVIDIA A100 (80GB)
- NVIDIA H100 (80GB)
- NVIDIA B200 (192GB)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import math


@dataclass
class HardwareSpec:
    name: str
    vram_gb: float
    mac_units: int
    clock_ghz: float
    bandwidth_gb_s: float
    network_type: str  # "tcp_asyncio" or "nvlink" or "tpu_ici"
    network_latency_us: float  # microseconds


@dataclass
class ModelSpec:
    name: str
    layers: int
    d_model: int
    layer_weight_gb: float  # BF16 layer weight size in GB
    layer_active_params: float  # Number of active params per layer
    kv_bytes_per_token_layer: float  # KV cache bytes per token in 1 layer
    context_window: int


HARDWARE_DATABASE: Dict[str, HardwareSpec] = {
    "NVIDIA T4": HardwareSpec(
        name="NVIDIA T4",
        vram_gb=16.0,
        mac_units=20_480,
        clock_ghz=1.59,
        bandwidth_gb_s=300.0,
        network_type="tcp_asyncio",
        network_latency_us=70.0,
    ),
    "NVIDIA L4": HardwareSpec(
        name="NVIDIA L4",
        vram_gb=24.0,
        mac_units=29_696,
        clock_ghz=2.04,
        bandwidth_gb_s=300.0,
        network_type="tcp_asyncio",
        network_latency_us=70.0,
    ),
    "Google TPU v4": HardwareSpec(
        name="Google TPU v4",
        vram_gb=32.0,
        mac_units=65_536,
        clock_ghz=1.05,
        bandwidth_gb_s=1200.0,
        network_type="tpu_ici",
        network_latency_us=2.0,
    ),
    "Google TPU v5p": HardwareSpec(
        name="Google TPU v5p",
        vram_gb=95.0,
        mac_units=131_072,
        clock_ghz=1.75,
        bandwidth_gb_s=4800.0,
        network_type="tpu_ici",
        network_latency_us=2.0,
    ),
    "NVIDIA A100 (80GB)": HardwareSpec(
        name="NVIDIA A100 (80GB)",
        vram_gb=80.0,
        mac_units=110_592,
        clock_ghz=1.41,
        bandwidth_gb_s=2039.0,
        network_type="nvlink",
        network_latency_us=1.5,
    ),
    "NVIDIA H100 (80GB)": HardwareSpec(
        name="NVIDIA H100 (80GB)",
        vram_gb=80.0,
        mac_units=270_336,
        clock_ghz=1.83,
        bandwidth_gb_s=3350.0,
        network_type="nvlink",
        network_latency_us=1.0,
    ),
    "NVIDIA B200 (192GB)": HardwareSpec(
        name="NVIDIA B200 (192GB)",
        vram_gb=192.0,
        mac_units=625_000,
        clock_ghz=1.80,
        bandwidth_gb_s=8000.0,
        network_type="nvlink",
        network_latency_us=0.8,
    ),
}

MODEL_DATABASE: Dict[str, ModelSpec] = {
    "Mixtral 8x7B": ModelSpec(
        name="Mixtral 8x7B",
        layers=32,
        d_model=4096,
        layer_weight_gb=2.90,  # ~1.45B params per layer * 2 bytes
        layer_active_params=376_000_000,  # top-2 active
        kv_bytes_per_token_layer=4096.0,
        context_window=32_768,
    ),
    "Llama 3.1 8B": ModelSpec(
        name="Llama 3.1 8B",
        layers=32,
        d_model=4096,
        layer_weight_gb=0.436,
        layer_active_params=218_000_000,
        kv_bytes_per_token_layer=4096.0,
        context_window=128_000,
    ),
    "Llama 3.1 70B": ModelSpec(
        name="Llama 3.1 70B",
        layers=80,
        d_model=8192,
        layer_weight_gb=1.71,
        layer_active_params=856_000_000,
        kv_bytes_per_token_layer=4096.0,
        context_window=128_000,
    ),
    "DeepSeek-V3 (671B)": ModelSpec(
        name="DeepSeek-V3 (671B)",
        layers=61,
        d_model=7168,
        layer_weight_gb=23.0,
        layer_active_params=585_000_000,
        kv_bytes_per_token_layer=1152.0,  # MLA compressed
        context_window=128_000,
    ),
    "Llama 3.1 405B": ModelSpec(
        name="Llama 3.1 405B",
        layers=126,
        d_model=16384,
        layer_weight_gb=6.38,
        layer_active_params=3_188_000_000,
        kv_bytes_per_token_layer=4096.0,
        context_window=128_000,
    ),
}


def simulate(
    model: ModelSpec,
    hw: HardwareSpec,
    seq_len: int,
    batch_size: int = 1,
    os_overhead_gb: float = 2.0,
) -> Dict[str, float]:
    """Calculates optimal partitioning and performance metrics for a single hardware type."""
    # Usable VRAM per node
    usable_vram_gb = max(0.0, hw.vram_gb - os_overhead_gb)

    # KV cache size per layer in GB
    kv_cache_layer_gb = (seq_len * model.kv_bytes_per_token_layer * batch_size) / (1024**3)

    # Total memory required for 1 full layer (Weights + KV Cache)
    total_mem_per_layer_gb = model.layer_weight_gb + kv_cache_layer_gb

    # Maximum layers that fit on 1 node
    if total_mem_per_layer_gb > usable_vram_gb:
        max_layers_per_node = 0
        nodes_required = math.inf
    else:
        max_layers_per_node = int(usable_vram_gb // total_mem_per_layer_gb)
        max_layers_per_node = min(max_layers_per_node, model.layers)
        nodes_required = math.ceil(model.layers / max_layers_per_node)

    if max_layers_per_node == 0:
        return {
            "model": model.name,
            "hw": hw.name,
            "fits": False,
            "layers_per_node": 0,
            "nodes_required": math.inf,
            "ms_per_token": math.inf,
            "tps": 0.0,
        }

    # Layers per node distribution
    # 1. Compute time per layer for batch_size (seconds)
    # Active FLOPs = 2 * active_params * batch_size
    active_macs_layer = model.layer_active_params * batch_size
    cycles_compute = active_macs_layer / hw.mac_units
    t_compute_layer_s = cycles_compute / (hw.clock_ghz * 1e9)

    # 2. Memory fetch time per layer (seconds)
    # In decode (batch_size=1), weights + KV cache must stream from VRAM to SRAM
    total_bytes_streamed = (model.layer_weight_gb + kv_cache_layer_gb) * (1024**3)
    t_memory_layer_s = total_bytes_streamed / (hw.bandwidth_gb_s * 1e9)

    # Real execution time per layer on this GPU (Roofline bottleneck)
    t_layer_s = max(t_compute_layer_s, t_memory_layer_s)

    # Total execution time across all model layers
    t_model_compute_s = t_layer_s * model.layers

    # 3. Network handoff time between nodes
    # Data sent between stages: [batch_size, d_model] * 2 bytes (BF16)
    packet_bytes = batch_size * model.d_model * 2
    # Number of network cuts = nodes_required - 1
    network_cuts = max(0, nodes_required - 1)
    t_network_per_cut_s = (hw.network_latency_us * 1e-6) + (packet_bytes / (3.125e9 if hw.network_type == "tcp_asyncio" else 50e9))
    t_total_network_s = network_cuts * t_network_per_cut_s

    # Total time per token (seconds)
    t_token_s = t_model_compute_s + t_total_network_s
    t_token_ms = t_token_s * 1000.0
    tps = 1.0 / t_token_s if t_token_s > 0 else 0.0

    return {
        "model": model.name,
        "hw": hw.name,
        "fits": True,
        "layers_per_node": max_layers_per_node,
        "nodes_required": nodes_required,
        "cuts": network_cuts,
        "bottleneck": "Memory (Bandwidth)" if t_memory_layer_s > t_compute_layer_s else "Compute (MACs)",
        "t_layer_ms": t_layer_s * 1000.0,
        "t_model_compute_ms": t_model_compute_s * 1000.0,
        "t_network_ms": t_total_network_s * 1000.0,
        "ms_per_token": t_token_ms,
        "tps": tps,
    }


def print_comparison(model_name: str, seq_len: int):
    model = MODEL_DATABASE[model_name]
    print(f"\n{'='*115}")
    print(f" MODEL: {model.name} ({model.layers} layers, d_model={model.d_model}) | PROMPT CONTEXT: {seq_len:,} tokens (BF16)")
    print(f"{'='*115}")
    print(
        f"{'Hardware':23s} | {'Layers/Node':11s} | {'Nodes Needed':12s} | {'Bottleneck':18s} | "
        f"{'Compute (ms)':12s} | {'Net Latency':11s} | {'Total ms/tok':12s} | {'TPS (tok/s)':11s}"
    )
    print(f"{'-'*115}")

    for hw in HARDWARE_DATABASE.values():
        res = simulate(model, hw, seq_len=seq_len)
        if not res["fits"]:
            print(f"{hw.name:23s} | {'0 (OOM)':11s} | {'N/A (Too Big)':12s} | {'VRAM Overflow':18s} | {'-':12s} | {'-':11s} | {'-':12s} | {'0.0':11s}")
        else:
            print(
                f"{hw.name:23s} | {res['layers_per_node']:11d} | {res['nodes_required']:12d} | "
                f"{res['bottleneck']:18s} | {res['t_model_compute_ms']:12.2f} | {res['t_network_ms']:11.3f} | "
                f"{res['ms_per_token']:12.2f} | {res['tps']:11.1f}"
            )


if __name__ == "__main__":
    # Compare Mixtral 8x7B (from cluster_config.json) at 4K and 32K
    print_comparison("Mixtral 8x7B", seq_len=4096)
    print_comparison("Mixtral 8x7B", seq_len=32768)

    # Compare Llama 3.1 70B at 4K and 128K
    print_comparison("Llama 3.1 70B", seq_len=4096)
    print_comparison("Llama 3.1 70B", seq_len=128000)
