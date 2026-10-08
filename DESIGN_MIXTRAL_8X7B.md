# FrontierSplit Architecture Specification: 8-Node Unquantized Mixtral 8x7B Cluster

**Model**: `mistralai/Mixtral-8x7B-Instruct-v0.1`  
**Precision**: **100% Unquantized FP16** (Zero quantization, full native precision)  
**Total Parameters**: 46.7 Billion  
**Active Parameters per Token**: 12.9 Billion (Top-2 Routed MoE)  
**Cluster Topology**: 8-Stage Pipeline Parallelism ($K=8$)  
**Target Hardware**: 8x GCP `n1-standard-8` (8 vCPU, 30 GB RAM, 1x NVIDIA Tesla T4 16 GB VRAM each = 128 GB Aggregate GPU VRAM)  
**Storage**: 200 GB Balanced Persistent Disk per node (accommodates 87 GB unquantized safetensors + OS)  
**Interconnect**: GCP Internal VPC Subnet (MTU 1460, $< 0.4\text{ ms}$ inter-node latency)

---

## 1. Executive Summary & Precision Invariant

The FrontierSplit architecture deploys **unquantized FP16** across commodity 16 GB GPUs. Rather than compressing weights into 4-bit/8-bit representations that compromise perplexity and reasoning rigor, FrontierSplit distributes the unquantized 86.99 GB model across 8 commodity Tesla T4 nodes using Pipeline Parallelism.

Because Mixtral 8x7B uses sparse Mixture-of-Experts routing:
- **Total Parameters**: 46.7B ($86.99\text{ GB}$ in FP16).
- **Active Parameters / Token**: 12.9B ($25.8\text{ GB}$ in FP16).
- Each token activates only 2 out of 8 experts per layer.
- Compute per token increases by only $\sim 1.78\times$ over Mistral 7B, while reasoning depth matches a 47B dense model.

---

## 2. Unquantized FP16 Memory & Partitioning Plan

Using the FrontierSplit Slicer profiler (`frontiersplit.slicer`), the 32 transformer layers of Mixtral 8x7B are split evenly across 8 nodes (**4 layers per node**).

| Node ID | Instance Name | Assigned Layers | Primary Role | Unquantized FP16 Weights | Clean VRAM Headroom | Max KV Cache Capacity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Node 1** | `frontiersplit-node-1` | **L00 – L03** | Ingress Gateway + Embeddings + Stage 0 | **11.06 GB** | **$\approx 4.3\text{ GB}$** | $> 100{,}000\text{ tokens}$ |
| **Node 2** | `frontiersplit-node-2` | **L04 – L07** | Pipeline Stage 1 Compute | **10.81 GB** | **$\approx 4.5\text{ GB}$** | $> 100{,}000\text{ tokens}$ |
| **Node 3** | `frontiersplit-node-3` | **L08 – L11** | Pipeline Stage 2 Compute | **10.81 GB** | **$\approx 4.5\text{ GB}$** | $> 100{,}000\text{ tokens}$ |
| **Node 4** | `frontiersplit-node-4` | **L12 – L15** | Pipeline Stage 3 Compute | **10.81 GB** | **$\approx 4.5\text{ GB}$** | $> 100{,}000\text{ tokens}$ |
| **Node 5** | `frontiersplit-node-5` | **L16 – L19** | Pipeline Stage 4 Compute | **10.81 GB** | **$\approx 4.5\text{ GB}$** | $> 100{,}000\text{ tokens}$ |
| **Node 6** | `frontiersplit-node-6` | **L20 – L23** | Pipeline Stage 5 Compute | **10.81 GB** | **$\approx 4.5\text{ GB}$** | $> 100{,}000\text{ tokens}$ |
| **Node 7** | `frontiersplit-node-7` | **L24 – L27** | Pipeline Stage 6 Compute | **10.81 GB** | **$\approx 4.5\text{ GB}$** | $> 100{,}000\text{ tokens}$ |
| **Node 8** | `frontiersplit-node-8` | **L28 – L31** | Stage 7 Compute + RMSNorm + LM Head | **11.06 GB** | **$\approx 4.3\text{ GB}$** | $> 100{,}000\text{ tokens}$ |

### VRAM Headroom & KV Cache Sizing
- 16 GB Tesla T4 physical capacity: $15{,}360\text{ MB}$.
- Weights footprint: $\sim 10{,}810\text{ MB} - 11{,}060\text{ MB}$.
- PyTorch CUDA context & buffers: $\approx 600\text{ MB}$.
- **Net Available VRAM for KV Caches**: $\approx 3{,}700\text{ MB} - 3{,}950\text{ MB}$ per node.
- KV Cache per token across 4 layers of GQA ($N_{kv}=8$, $d_{head}=128$, FP16):
  $$4 \text{ layers} \times 2 \times 8 \times 128 \times 2\text{ bytes} = 16{,}384\text{ bytes} = 16\text{ KB / token}$$
  $$\text{KV Token Capacity} = \frac{3{,}700\text{ MB}}{16\text{ KB}} \approx 230{,}000\text{ tokens}$$
This headroom allows deep multi-turn chat sessions and concurrent streams without any VRAM exhaustion.

---

## 3. Batched GEMM Decode Transferability

### 3.1 100% Shared Attention Architecture
Mixtral 8x7B retains the exact Grouped-Query Attention (GQA) geometry of Mistral 7B:
$$\text{hidden\_size} = 4096, \quad N_{\text{heads}} = 32, \quad N_{\text{kv\_heads}} = 8, \quad d_{\text{head}} = 128$$

1. **Rotary Position Embeddings (RoPE)**: Direct 1:1 mapping with identical sinusoidal frequencies.
2. **Fused QKV Projection**: Linear projections across the batched decode dimension $[B, 1, 4096]$ remain single dense GEMMs.
3. **Contiguous KV Cache**: Dynamic per-session KV caches maintain identical dimensions $[B, 8, L, 128]$.
4. **PyTorch SDPA**: Native FlashAttention/SDPA kernels execute seamlessly across sessions.

### 3.2 Wire Activation Invariant
The inter-stage activation tensor passed via HTTP/gRPC is:
$$\text{Shape: } [B, 1, 4096] \quad (\approx 64\text{ KB in FP16 for } B=8)$$

Across 7 inter-node hops ($8 \text{ nodes} - 1$):
$$\text{Total Network Latency per Decode Step} = 7 \times 0.4\text{ ms} \approx 2.8\text{ ms}$$
This represents $< 1.5\%$ of the total step latency budget ($\sim 250\text{ ms}$), proving that scaling pipeline depth to 8 stages incurs virtually zero network tax.

### 3.3 Sparse MoE Feed-Forward Layer Handling
In `frontiersplit/worker.py`, the feed-forward step automatically checks for the MoE block:
```python
# Normed hidden states: [B, 1, 4096]
normed = layer.post_attention_layernorm(hidden_states)

if hasattr(layer, "block_sparse_moe"):
    # Hugging Face MixtralSparseMoeBlock:
    # 1. Router Gate W_gate computes logits [B, 8]
    # 2. Top-2 softmax selects top experts per token
    # 3. Batched expert dispatch groups tokens routed to the same expert
    moe_out = layer.block_sparse_moe(normed)
    moe_out = moe_out[0] if isinstance(moe_out, tuple) else moe_out
    hidden_states = hidden_states + moe_out
elif hasattr(layer, "mlp"):
    # Dense Mistral MLP fallback
    hidden_states = hidden_states + layer.mlp(normed)
```

---

## 4. Host Hardware & Operational Cost

| Parameter | Configuration | Rationale |
| :--- | :--- | :--- |
| **Machine Type** | `n1-standard-8` (8 vCPU, 30 GB RAM) | Ample host RAM for streaming 4.9 GB safetensors shards |
| **Accelerator** | 1x NVIDIA Tesla T4 (16 GB GDDR6) | 128 GB aggregate VRAM across 8 nodes |
| **Disk Size** | `200GB pd-balanced` | Comfortably holds full 87 GB unquantized checkpoint + OS |
| **Compute Cost** | $\$0.47 / \text{hr}$ per node ($\$3.76 / \text{hr}$ cluster) | Only active when cluster is running |
| **Stopped Cost** | **$\$0.00 / \text{hr}$ compute** | Disks only ($\approx \$1.10 / \text{day}$ aggregate across all 8 nodes) |

---

## 5. Verification & Deployment Steps

1. **Environment Configuration**: Set `NUM_NODES=8`, `MODEL_ID="mistralai/Mixtral-8x7B-Instruct-v0.1"`, `MACHINE_TYPE="n1-standard-8"`, `DISK_SIZE="200GB"` in `scripts/env.sh`.
2. **Cluster Topology Definition**: Update `cluster_config.json` with 8 nodes and exact layer ranges (`[0, 3]`, `[4, 7]`, ..., `[28, 31]`).
3. **Cluster Up & Provisioning**: Run `./scripts/cluster_up.sh` to provision the 8 instances.
4. **Pipeline Ignition**: Run `./scripts/start_pipeline.sh` to initialize workers in reverse stage order (Stage 7 $\to$ Stage 0) and launch the Gateway.
5. **Validation**: Execute `./scripts/cluster_check.sh` to verify end-to-end inference and latency across all 8 stages.
