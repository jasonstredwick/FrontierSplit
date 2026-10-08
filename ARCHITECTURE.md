# FrontierSplit: Architecture Specification

## 1. System Overview

FrontierSplit is a distributed inference architecture designed to run **unquantized (FP16/FP8) Frontier Mixture-of-Experts (MoE) models** across clusters of commodity, non-NVLink compute nodes.

Traditional serving engines (vLLM, TensorRT-LLM) rely on **Tensor Parallelism (TP)** and **Expert Parallelism (EP)**, which demand high-bandwidth, ultra-low-latency inter-GPU links (NVLink / InfiniBand / TPU optical interconnects) to handle frequent all-to-all scatter/gather communications.

FrontierSplit decouples the model vertically by **Pipeline Parallelism (PP)**:
* Each compute node hosts a contiguous slice of transformer layers (e.g., Node 1 hosts Layers 0–7, Node 2 hosts Layers 8–15, etc.).
* Across layer boundaries, nodes only exchange the **activation vector** for a sequence (a compact tensor of roughly 16 KB per token).
* This compact payload easily streams over standard datacenter **VPC Ethernet** (10–32 Gbps) with negligible latency overhead.
* The classic pipeline idle bubble is neutralized by feeding the pipeline with **concurrent multi-agent / multi-stream workloads**.

```
[ Client / Multi-Agent Harness (DSPy / OpenHands) ]
                     │ (Concurrent Token Streams)
                     ▼
         ┌─────────────────────────┐
         │ FrontierSplit Ingress   │ (OpenAI-compatible HTTP/gRPC)
         └───────────┬─────────────┘
                     │ Activation Tensor (16 KB / token)
                     ▼
         ┌─────────────────────────┐
         │ Node 0 (L4 GPU #1)      │ Layers 0 – 7 (Embeddings + MoE Layers)
         └───────────┬─────────────┘
                     │ Activation Tensor (VPC Network)
                     ▼
         ┌─────────────────────────┐
         │ Node 1 (L4 GPU #2)      │ Layers 8 – 15
         └───────────┬─────────────┘
                     │ Activation Tensor (VPC Network)
                     ▼
         ┌─────────────────────────┐
         │ Node 2 (L4 GPU #3)      │ Layers 16 – 23
         └───────────┬─────────────┘
                     │ Activation Tensor (VPC Network)
                     ▼
         ┌─────────────────────────┐
         │ Node 3 (L4 GPU #4)      │ Layers 24 – 31 (MoE Layers + LM Head)
         └───────────┬─────────────┘
                     │ Generated Token
                     ▼
         [ Response Stream / KV Feedback ]
```

---

## 2. Hardware Topology

### Baseline Target: 4x NVIDIA L4 Cluster (GCP G2 Instances)
* **Instance Type**: 4x `g2-standard-4` (or `g2-standard-8`)
* **GPU**: 1x NVIDIA L4 per node (24 GB GDDR6 VRAM per GPU)
* **Total Cluster VRAM**: 96 GB VRAM
* **Network**: Google Cloud VPC in a single region/zone (`us-central1-a`)
* **Egress Cost**: $0.00 internal zone egress
* **Target Model**: Mixtral 8x7B (FP16 ~94 GB, or FP8 ~48 GB)

---

## 3. Core Software Components

1. **Model Slicer & Memory Planner (`frontiersplit/slicer.py`)**:
   * Inspects HuggingFace `config.json` for target models.
   * Computes per-layer parameter footprint ($W \times H$ matrices for attention, shared experts, and routed experts).
   * Determines optimal layer distribution across available node VRAM.

2. **Transport & Pipeline Worker (`frontiersplit/worker.py`)**:
   * Runs on each cluster node.
   * Holds allocated model layers in local GPU VRAM.
   * Receives incoming activation tensors over gRPC / raw TCP sockets.
   * Computes attention + MoE feed-forward passes for its assigned layers.
   * Forwards the output tensor to the next downstream node.

3. **Ingress Coordinator & API Gateway (`frontiersplit/gateway.py`)**:
   * Exposes standard OpenAI-compatible endpoints (`/v1/chat/completions`).
   * Manages request queuing, pipeline scheduling, and KV cache coordination.
   * Orchestrates token generation and autoregressive feedback loops.

4. **Multi-Agent Evaluation Harness (`benchmarks/`)**:
   * Drives heavy concurrency into the pipeline to eliminate pipeline bubbles.
   * Integrates benchmarks: **SWE-bench Lite** and **IFEval**.
   * Telemetry tracker measuring GPU compute and VRAM saturation across all nodes.

---

## 4. KV Cache & Execution Architecture

For complete design details, memory layouts, and algorithmic proofs, see [DESIGN_KV_CACHE.md](file:///Users/pixel/code/FrontierSplit/DESIGN_KV_CACHE.md).

* **Mode**: Bounded Tile-Aligned KV Cache with Geometric Doubling.
* **Concurrency**: Multi-Producer API Layer (FastAPI) feeding a Single-Consumer Execution Engine Thread (Zero GPU mutexes).
* **Hardware Alignment**: All sequence allocations are aligned to 16-token boundaries ($\operatorname{ceil}_{16}$), matching the $16 \times 16 \times 16$ Tensor Core tile and 32-thread warps.
* **Allocation Policy**:
  * If $\text{hard\_cap} \le 256$: Allocate $\operatorname{ceil}_{16}(\text{hard\_cap})$ directly (zero memory waste, zero reallocations).
  * If $\text{hard\_cap} > 256$: Allocate to nearest power-of-two for initial expected response (~256 tokens), doubling geometrically ($256 \to 512 \to 1024 \dots$) if capacity is exhausted.
* **Memory Invariant**: Memory remains 100% physically contiguous, enabling standard cuBLAS / PyTorch GEMM execution without custom non-contiguous paging kernels.
* **Pipeline Boundary Invariant**: Node 1 and Node 2 maintain independent local session caches; only the transient `[M, 4096]` activation tensor crosses the VPC network wire.

