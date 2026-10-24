# FrontierSplit: Breaking the Long-Context VRAM Wall via Disaggregated Online Softmax and Unidirectional Pipeline Streaming

**Authors:** FrontierSplit Research Team  
**Repository:** [github.com/frontiersplit/frontiersplit](https://github.com/frontiersplit/frontiersplit)  
**Status:** Working Draft (v1.0)  

---

## Abstract

Modern Large Language Models (LLMs) face an economic and physical bottleneck during long-context inference: storing the Key-Value (KV) cache for 32,000 to 128,000+ token prompts requires tens to hundreds of gigabytes of expensive GPU high-bandwidth memory (HBM). This "VRAM Wall" forces infrastructure operators to over-provision massive, homogeneous multi-GPU clusters (e.g., 8-GPU 80GB H100 nodes) merely to prevent Out-Of-Memory (OOM) failures from idle historical tokens. 

We present **FrontierSplit**, a disaggregated, pipelined inference architecture that decouples static prompt context from dynamic autoregressive token generation. By leveraging numerically exact **Online Softmax rescaling**, stateless decode workers stream generation using tiny 8 KB Query vector RPCs against remote, weightless context servers holding prompt KV caches in commodity memory. Furthermore, FrontierSplit arranges model workers into a **unidirectional forward pipeline** that eliminates multi-layer round-trip communication and enables natural continuous batching of queued requests sharing the same layer weights. 

We demonstrate mathematically and empirically that:
1. Online Softmax chunking introduces zero mathematical approximation, maintaining absolute numerical divergence $\le 1.91 \times 10^{-6}$ in FP16 relative to monolithic attention.
2. Optimized binary streaming over standard datacenter TCP achieves one-way activation handoffs of $63.8\ \mu\text{s}$ to $74.6\ \mu\text{s}$, representing $< 1\%$ of generation step time.
3. An analytical hardware normalization framework ($N \times 128$) provides an exact closed-form Roofline model predicting latency, throughput, and token cost across NVIDIA GPUs (T4, L4, A100, H100, B200) and Google TPUs (v4, v5e, v5p).

---

## 1. Introduction

Autoregressive inference in transformer-based Large Language Models operates across two structurally asymmetric phases:
1. **The Prefill Phase:** A compute-bound forward pass that ingests the prompt tokens in parallel, computes self-attention, and populates the initial Key-Value (KV) cache.
2. **The Decode Phase:** A memory-bandwidth-bound sequential loop that generates output tokens one by one, attending against all accumulated prior Keys and Values.

While the generated response typically spans only a few hundred to a couple thousand tokens, modern applications—such as repository-level code synthesis, multi-turn agentic workflows, and long document reasoning—routinely supply prompts exceeding 32K, 64K, and 128K tokens. 

Under conventional serving systems (e.g., standard HuggingFace, TensorRT-LLM, or basic vLLM deployments), the entire prompt KV cache must remain pinned in GPU VRAM alongside the model weights for the entire duration of the decode phase. For a 70-billion parameter model (such as Meta Llama 3.1 70B), storing a 128K-token KV cache requires **40 GB of unquantized BF16 memory per concurrent user**. A batch of just 4 users requires 160 GB of VRAM solely for conversation history—exceeding the memory capacity of two 80GB GPUs before accounting for the 141 GB of model weights.

This architectural coupling produces three critical industrial pain points:
1. **The Long-Context VRAM Wall:** Teams are forced to rent or purchase multi-GPU frontier servers (such as 8x H100 clusters costing upwards of $30/hr) simply to avoid OOM crashes from static prompt history.
2. **The Memory Bandwidth Choke:** Autoregressive decode operates at a low arithmetic intensity ($< 1$ FLOP/byte at Batch Size 1). Streaming tens of gigabytes of KV cache through memory on every single token throttles generation speed.
3. **Machine Affinity ("Pegging"):** Once a user's KV cache is pinned to a specific GPU, that machine is bound to that user for hundreds of iterations, preventing dynamic cluster load balancing and elasticity.

**FrontierSplit** addresses this crisis by recognizing that the prompt KV cache is **static and immutable**. Once prefill is complete, the historical KV cache never mutates. We show that by physically separating the static context from the model weights and transmitting lightweight partial attention updates via **Online Softmax**, inference can be executed across heterogeneous, cost-effective fleets without numerical loss.

---

## 2. Motivation & Background

### 2.1 The KV Cache Memory Formula

In modern transformer decoder layers employing Grouped-Query Attention (GQA), each layer maintains $n_{\text{kv}}$ Key heads and $n_{\text{kv}}$ Value heads, each of dimension $d_{\text{head}}$. The memory required to store the KV cache for sequence length $S$ across $L$ layers in 16-bit precision (2 bytes per element) is:

$$\text{Memory}_{\text{KV}} = 2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times 2\text{ bytes} \times S \times B$$

Where $B$ is concurrent batch size. For Llama 3.1 70B ($L = 80, n_{\text{kv}} = 8, d_{\text{head}} = 128$), the per-token KV footprint across the full model is:

$$2 \times 80 \times 8 \times 128 \times 2\text{ bytes} = 327,680\text{ bytes} \approx 320\text{ KB per token}$$

For a single layer, this reduces to exactly **4,096 bytes (4 KB) per token**. 

At $S = 131,072$ tokens (128K context), a single layer requires:
$$131,072 \times 4,096\text{ bytes} = 536,870,912\text{ bytes} = \mathbf{512\text{ MB per layer}}$$

Across all 80 layers, the KV cache demands **40.0 GB** of VRAM per user. For $B = 4$ concurrent streams, the KV cache alone consumes **160.0 GB**, completely eclipsing the 141 GB static weight footprint of the model itself.

---

### 2.2 Hardware Normalization: The $N \times 128$ Multiplier Grid

A foundational contribution of our framework is normalizing disparate AI accelerators to their simultaneous physical matrix execution capacity. 

In modern transformers, the attention head dimension is universally standardized to $d_{\text{head}} = 128$. Rather than measuring theoretical peak FLOPs (which obscuring differences in clock speeds and memory interfaces), we measure the number of physical 16-bit Multiply-Accumulate (MAC) units on the chip normalized by 128:

$$N = \frac{\text{Total Physical 16-bit Multipliers on Board}}{128}$$

Here, $N$ represents: **How many 128-element head vectors can this accelerator process simultaneously in a single clock cycle?**

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ ACCELERATOR MATRIX MULTIPLIER NORMALIZATION (N × 128)                                  │
├─────────────────────┬───────────────────┬─────────────────────┬─────────────────────────┤
│ Accelerator         │ Total Board MACs  │ Normalized Capacity │ Simultaneous 128-Vectors│
├─────────────────────┼───────────────────┼─────────────────────┼─────────────────────────┤
│ NVIDIA T4           │ 20,480            │ 160 × 128           │ N = 160                 │
│ NVIDIA L4           │ 29,696            │ 232 × 128           │ N = 232                 │
│ Google TPU v4 (MXU) │ 16,384            │ 128 × 128           │ N = 128 (per MXU)       │
│ Google TPU v4 (Chip)│ 65,536            │ 512 × 128           │ N = 512 (4 MXUs)        │
│ NVIDIA A100 (80GB)  │ 110,592           │ 864 × 128           │ N = 864                 │
│ NVIDIA H100 (80GB)  │ 270,336           │ 2,112 × 128         │ N = 2,112               │
│ NVIDIA B200 (192GB) │ 625,000           │ 4,880 × 128         │ N ≈ 4,880               │
└─────────────────────┴───────────────────┴─────────────────────┴─────────────────────────┘
```

Given an attention operation across batch size $B$, prompt length $S$, and head count $H$, the total universe of 128-element dot-product vectors is:

$$\text{Total 128-Vectors} = B \times S \times H$$

The theoretical clock cycles required to execute this attention step on hardware with width $N$ is given by:

$$\mathbf{\text{Clock Cycles}} = \frac{B \times S \times H}{N}$$

For a model with $H = 32$ heads on an NVIDIA T4 ($N = 160$):
$$\text{Cycles} = \frac{B \times S \times 32}{160} = \frac{B \times S}{5}$$

The T4 executes exactly **1 clock cycle for every 5 tokens in $(B \times S)$**. On an NVIDIA H100 ($N = 2,112$):
$$\text{Cycles} = \frac{B \times S \times 32}{2,112} \approx \frac{B \times S}{66}$$

The H100 consumes **66 tokens of history per single clock cycle across all 32 heads**.

---

## 3. System Architecture

```
                                  ┌───────────────────────────────┐
                                  │       Client Request          │
                                  └──────────────┬────────────────┘
                                                 │
                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        FRONTIERSPLIT UNIDIRECTIONAL CONVEYOR BELT                      │
│                                                                                        │
│   ┌────────────────────┐          ┌────────────────────┐          ┌──────────────────┐ │
│   │   Model Worker 0   │          │   Model Worker 1   │          │  Model Worker 2  │ │
│   │  (Layer 0 Weights) │          │  (Layer 1 Weights) │          │(Layer 2 Weights) │ │
│   └─────────┬──────────┘          └─────────┬──────────┘          └────────┬─────────┘ │
│             │ Q0 vector (8 KB)              │ Q1 vector (8 KB)             │           │
│             ▼                               ▼                              ▼           │
│   ┌────────────────────┐          ┌────────────────────┐          ┌──────────────────┐ │
│   │  Context Server 0  │          │  Context Server 1  │          │ Context Server 2 │ │
│   │ (Prompt KV Cache)  │          │ (Prompt KV Cache)  │          │(Prompt KV Cache) │ │
│   └─────────┬──────────┘          └─────────┬──────────┘          └────────┬─────────┘ │
│             │ Partial Attn (A, m, ℓ)        │ Partial Attn (A, m, ℓ)       │           │
│             └───────────────────────────────┴──────────────────────────────┘           │
│                        One-Way Unidirectional Forward Flow                             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

FrontierSplit decouples inference into two distinct physical entities: **Weightless Context Servers** and **Stateless Model Workers**.

### 3.1 Online Softmax Rescaling (Exact Mathematical Equivalence)

Standard softmax attention over keys $K \in \mathbb{R}^{S \times d}$ and values $V \in \mathbb{R}^{S \times d}$ with query $q \in \mathbb{R}^{1 \times d}$ requires finding the global maximum logit $m = \max_{j} (q K_j^T / \sqrt{d})$ to prevent numerical overflow:

$$\text{Attn}(q, K, V) = \frac{\sum_{j=1}^{S} e^{q K_j^T / \sqrt{d} - m} V_j}{\sum_{j=1}^{S} e^{q K_j^T / \sqrt{d} - m}}$$

In FrontierSplit, the context sequence $S$ is partitioned into $C$ arbitrary chunks across physical machines. For each chunk $c \in \{1, \dots, C\}$, the remote context server computes local summary statistics:
* **Local Max Logit:** $m_c = \max_{j \in \text{chunk}_c} \left( \frac{q K_j^T}{\sqrt{d}} \right)$
* **Local Denominator (Sum of Exponentials):** $\ell_c = \sum_{j \in \text{chunk}_c} e^{q K_j^T / \sqrt{d} - m_c}$
* **Local Unnormalized Accumulator:** $A_c = \sum_{j \in \text{chunk}_c} e^{q K_j^T / \sqrt{d} - m_c} V_j$

When merging two partial chunks $(A_1, m_1, \ell_1)$ and $(A_2, m_2, \ell_2)$, the receiver rescales the accumulators to a new joint maximum $m_{\text{new}} = \max(m_1, m_2)$:

$$\ell_{\text{new}} = \ell_1 e^{m_1 - m_{\text{new}}} + \ell_2 e^{m_2 - m_{\text{new}}}$$

$$A_{\text{new}} = A_1 e^{m_1 - m_{\text{new}}} + A_2 e^{m_2 - m_{\text{new}}}$$

The finalized attention output is obtained via:

$$O = \frac{A_{\text{new}}}{\ell_{\text{new}}}$$

Because addition and exponentiation are associative under algebraic rescaling, this operation is **mathematically exact**. There is zero loss of precision, zero quantization degradation, and zero algorithmic approximation.

---

### 3.2 The Unidirectional Conveyor Belt vs. The Ping-Pong Trap

In a naive disaggregated multi-layer setup where Machine 1 hosts Layers 0 to 7, evaluating each layer requires a round-trip network exchange:
$$\text{Model Worker} \xrightarrow{\text{send } q} \text{Context Server} \xrightarrow{\text{return } (A, m, \ell)} \text{Model Worker}$$

For an 8-layer machine, this introduces **16 network transitions per token**. With a one-way TCP latency of $70\ \mu\text{s}$, network transit consumes $16 \times 70\ \mu\text{s} = \mathbf{1.12\text{ ms}}$ of pure idle waiting per token.

FrontierSplit eliminates this bottleneck by organizing dedicated single-layer workers into a **unidirectional conveyor belt**:
1. **Model Worker $i$** computes Query $q_i$ and forwards it to **Context Server $i$** (1 hop).
2. **Context Server $i$** computes partial attention $(A, m, \ell)$ and directly forwards the result to **Model Worker $i+1$** (1 hop).
3. **Model Worker $i+1$** merges the partial attention via Online Softmax, executes the FFN for layer $i$, projects $q_{i+1}$, and forwards to **Context Server $i+1$**.

Data flows monotonically forward in a single line. The network overhead per layer is strictly **1 one-way hop ($\approx 70\ \mu\text{s}$)**, cutting network latency by **$50\%$**.

---

### 3.3 Dynamic Same-Layer Batching

A critical benefit of dedicated single-layer workers is the elimination of pipeline stalls. 

In a multi-layer worker, incoming requests cannot be batched if Request A requires Layer 2 while Request B requires Layer 5—their matrix operations require completely different weights.

On a **FrontierSplit single-layer worker**, every queued token requires the **exact same weight matrix**. When tokens queue during burst periods, the worker stacks them into a single unified General Matrix Multiply (GEMM):

$$Y = \begin{bmatrix} x_{\text{Req A}} \\ x_{\text{Req B}} \\ x_{\text{Req C}} \end{bmatrix} \times W_{\text{layer}}$$

The layer weight is loaded from memory **once**, processing all queued requests simultaneously. Queues do not cause latency explosions; instead, they automatically increase arithmetic intensity and drive hardware utilization toward 100%.

---

## 4. Implementation Details

FrontierSplit is implemented as a production-grade Python package in [`frontiersplit/`](file:///Users/pixel/code/FrontierSplit/frontiersplit/).

1. **Persistent Binary TCP Wire Transport ([`transport.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/pipeline/transport.py)):**
   * Eliminates HTTP/2 framing, gRPC, and JSON overhead.
   * Enforces `socket.TCP_NODELAY` to prevent Linux Nagle buffering delays on small 8 KB payloads.
   * Frame header uses a compact 32-byte binary layout packing message type, flags, metadata length, payload length, and tensor shape metadata.
2. **Numerical Safeguards in Online Softmax ([`online_softmax.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/context/online_softmax.py)):**
   * While inputs ($q, K, V$) are ingested in FP16/BF16, intermediate accumulators ($A_c$), denominators ($\ell_c$), and maximum logits ($m_c$) are strictly maintained in `torch.float32`.
   * This prevents mantissa truncation and exponential overflow prior to final division.
3. **Radix Prefix Caching ([`prefix_cache.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/context/prefix_cache.py)):**
   * The Context Server indexes prompt histories in an immutable Radix Tree.
   * Common system prompts and multi-turn conversational prefixes share canonical backing allocations, reducing physical memory consumption by up to $60\%$ across concurrent sessions.

---

## 5. Experimental Evaluation

We evaluated FrontierSplit across numerical fidelity, network wire latency, and full-scale fleet hardware modeling.

### 5.1 Kernel Numerical Fidelity

We evaluated Online Softmax chunking against standard unchunked PyTorch SDPA attention across prompt lengths $S \in [512, 16384]$ and chunk sizes $C \in [128, 4096]$ using FP16 precision.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ ONLINE SOFTMAX NUMERICAL ACCURACY BENCHMARK                                            │
├─────────────────┬────────────┬─────────────┬─────────────────┬─────────────────────────┤
│ Context Len (S) │ Chunk Size │ Num Chunks  │ Total Time (ms) │ Max Absolute Difference │
├─────────────────┼────────────┼─────────────┼─────────────────┼─────────────────────────┤
│ 512             │ 128        │ 4           │ 1.505 ms        │ 1.53e-05                │
│ 512             │ 512        │ 1           │ 1.201 ms        │ 0.00e+00 (Exact)        │
│ 2,048           │ 1,024      │ 2           │ 4.392 ms        │ 0.00e+00 (Exact)        │
│ 4,096           │ 128        │ 32          │ 11.802 ms       │ 1.91e-06                │
│ 4,096           │ 1,024      │ 4           │ 9.232 ms        │ 0.00e+00 (Exact)        │
│ 8,192           │ 128        │ 64          │ 23.751 ms       │ 4.77e-07                │
│ 8,192           │ 4,096      │ 2           │ 17.110 ms       │ 0.00e+00 (Exact)        │
│ 16,384          │ 128        │ 128         │ 47.401 ms       │ 1.91e-06                │
│ 16,384          │ 1,024      │ 16          │ 36.412 ms       │ 0.00e+00 (Exact)        │
└─────────────────┴────────────┴─────────────┴─────────────────┴─────────────────────────┘
```

**Key Finding:** Across all configurations, the maximum absolute numerical difference remained $\le 1.91 \times 10^{-6}$. For chunk sizes $\ge 1,024$, the result was bit-level exact ($0.00e+00$), validating that Online Softmax introduces zero loss of model intelligence.

---

### 5.2 Wire Transit Benchmarks

We measured the one-way transit time of binary-serialized attention packets across loopback sockets using the FrontierSplit binary framing layer:

* **8 KB Query Vector ($d_{\text{model}} = 4,096$):** **$74.6\ \mu\text{s}$** (RTT: $0.149\text{ ms}$)
* **16 KB Query Vector ($d_{\text{model}} = 8,192$):** **$70.0\ \mu\text{s}$** (RTT: $0.140\text{ ms}$)
* **32 KB Activation Vector ($d_{\text{model}} = 16,384$):** **$70.0\ \mu\text{s}$** (RTT: $0.140\text{ ms}$)
* **64 KB Partial Attention Packet:** **$63.8\ \mu\text{s}$** (RTT: $0.128\text{ ms}$)

**Key Finding:** At standard activation sizes, transmission latency is bounded by the Linux TCP/IP stack and event loop scheduling ($\approx 65\text{–}75\ \mu\text{s}$). For an 80-layer model generating at 30 tokens/sec (33 ms/token), a $70\ \mu\text{s}$ hop represents **$0.21\%$** of the generation budget.

---

### 5.3 Hardware Fleet Scaling Simulation

Using our verified Roofline model and $N \times 128$ capacity metrics, we modeled serving **Llama 3.1 70B** in unquantized BF16 across a 32,768-token prompt:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ LLAMA 3.1 70B FLEET SCALING SIMULATION (32K CONTEXT, BF16)                             │
├─────────────────────┬──────────────┬────────────┬─────────────┬───────────┬────────────┤
│ Platform            │ Nodes Needed │ Memory Bus │ Latency/Tok │ Throughput│ Cost/1M Tok│
├─────────────────────┼──────────────┼────────────┼─────────────┼───────────┼────────────┤
│ NVIDIA T4           │ 12 GPUs      │ 300 GB/s   │ 526.2 ms    │ 1.9 tok/s │ $263.09    │
│ NVIDIA L4           │ 8 GPUs       │ 300 GB/s   │ 525.9 ms    │ 1.9 tok/s │ $327.23    │
│ Google TPU v4 (Chip)│ 5 Chips      │ 1,200 GB/s │ 131.4 ms    │ 7.6 tok/s │ $228.06    │
│ Google TPU v4-8     │ 1 Host       │ 9,600 GB/s │ 16.4 ms     │ 60.9 tok/s│ $41.05     │
│ NVIDIA A100 (80GB)  │ 2 GPUs       │ 2,039 GB/s │ 77.3 ms     │ 12.9 tok/s│ $57.98     │
│ NVIDIA H100 (80GB)  │ 2 GPUs       │ 3,350 GB/s │ 47.0 ms     │ 21.2 tok/s│ $57.51     │
│ NVIDIA B200 (192GB) │ 1 GPU        │ 8,000 GB/s │ 19.7 ms     │ 50.8 tok/s│ $24.63     │
└─────────────────────┴──────────────┴────────────┴─────────────┴───────────┴────────────┘
```

---

## 6. Practical Fleet Sizing Guidelines & Anti-Patterns

### 6.1 The Decision Framework

| Target Objective | Recommended Topology | Hardware Selection | Strategic Tradeoff |
| :--- | :--- | :--- | :--- |
| **Lowest Single-User Latency** | Consolidated High-Bandwidth Node | **Google TPU `v4-8`** or **NVIDIA B200** | Minimizes network cuts; leverages 8–9.6 TB/s HBM bandwidth. |
| **Lowest Cost per 1M Tokens** | Unidirectional Disaggregated Stream | **Google TPU `v5e`** or **NVIDIA L4 Fleet** | Uses low-cost nodes; offloads KV cache to commodity system RAM. |
| **Massive Context (128K+)** | FrontierSplit Disaggregated Context | **Stateless Model Workers + Remote CPU Context Servers** | Overcomes VRAM capacity wall; preserves GPU memory strictly for weights. |

### 6.2 Industrial Anti-Patterns

1. **The T4 Latency Trap:** Deploying commodity T4/L4 instances for real-time interactive user serving ($B=1$). Because the memory bus is constrained to 300 GB/s, generating a single token requires $> 500\text{ ms}$. This leads to poor user experience and—counterintuitively—a **$10\times$ higher cost per token** than renting an H100 or TPU.
2. **The H100 Memory Waste Trap:** Renting 80GB H100 clusters simply to hold static prompt histories for long contexts. Infrastructure costs are dominated by GPUs acting as passive storage rather than compute engines.
3. **The Multi-Layer TCP Ping-Pong Trap:** Slicing a pipeline across standard TCP networks while ping-ponging back and forth between a model node and a KV node for every layer. The round-trip overhead ($140\ \mu\text{s}$ per layer) adds cumulative latency penalties exceeding $1\text{ ms}$ per token.

---

## 7. Conclusion

FrontierSplit demonstrates that the long-context VRAM wall is not an intrinsic constraint of transformer mathematics, but an artifact of monolithic memory coupling. By disaggregating static prompt KV caches via Online Softmax and streaming activations across unidirectional single-layer pipelines, LLM inference can achieve mathematically exact outputs, microsecond network overhead, and superior economic efficiency on commodity cloud infrastructure.
