# FrontierSplit: Hardware-Aligned Disaggregated Attention, Unidirectional Pipeline Streaming, and Fleet Optimization for Long-Context LLMs

**Authors:** FrontierSplit Research Team  
**Repository:** [github.com/frontiersplit/frontiersplit](https://github.com/frontiersplit/frontiersplit)  
**Status:** Comprehensive Technical Specification & Working Paper Draft (v2.0)

---

## Abstract

Serving unquantized Large Language Models (LLMs) at long context lengths ($S = 32\text{K to } 1\text{M+}$ tokens) exposes a fundamental conflict between **static model weights** and **dynamic Key-Value (KV) caches**. While model weights are identical across all concurrent users and dominated ($81\%$ to $98.5\%$) by post-attention Feed-Forward Networks (FFNs) or Mixture-of-Experts (MoE) layers, the KV cache is private to each conversation, grows linearly with sequence length ($1\text{ GiB}$ per $256\text{K}$ tokens per layer in standard GQA), and forces memory bandwidth bound execution. Co-locating both on monolithic GPU clusters creates the **Long-Context VRAM Wall**, **Machine Affinity ("Pegging")**, and **Multi-User Straggler Stalls**.

We present **FrontierSplit**, an analytical hardware-modeling framework and disaggregated inference architecture built on three core principles:
1. **Compute-to-Data Disaggregation via Online Softmax:** Instead of transferring multi-gigabyte KV caches across the network (Data-to-Compute), workers transmit tiny `8–32 KB` activation/query vectors (`~60–80 us` over persistent binary `asyncio` TCP) and merge arbitrarily partitioned KV chunks using exact **Online Softmax rescaling** in FP32.
2. **Intra-Layer Structural Decoupling & Unidirectional Streaming:** We analyze three distributed layer topologies—*Out-and-Back*, *Half-Layer Staggered*, and *Co-Located Attention+KV $\to$ Stateless FFN*. By recognizing that pre-attention and attention projections ($W_q, W_k, W_v, W_o$) represent only $1.5\%$ to $18\%$ of a layer's weights (`80–357 MB`), co-locating them directly on the KV cache node allows new $K, V$ tokens to be generated and chunked locally while streaming a single `8–32 KB` vector forward to **100% stateless FFN/MoE nodes**.
3. **Three-Tier Hardware-Aligned Chunking & Fleet Optimization:** By normalizing AI accelerators to their simultaneous 128-wide vector capacity ($N \times 128$) and modeling the $C$-cycle VRAM-to-SRAM double-buffering window alongside segmented memory partitions, we establish a **Three-Tier Chunking hierarchy** (Token/Warp Tile $\to$ SM/TensorCore Interruption Quantum $\to$ Multi-Node Online Softmax) and an objective **Fleet Sizing Methodology** that identifies when to consolidate, when to pipeline, and when to disaggregate across NVIDIA GPUs (T4, L4, A100, H100, B200) and Google Cloud TPUs (v4, v5e, v5p).

---

## 1. Introduction & Motivation

### 1.1 The Asymmetric Reality of Transformer Inference

Autoregressive generation in decoder-only transformers consists of two distinct operational regimes:
* **Prefill (Prompt Ingestion):** Processes all $S$ input tokens in parallel, computing initial self-attention and populating the Key-Value (KV) cache across all $L$ layers.
* **Decode (Token Generation):** Generates output tokens sequentially ($1$ token per step per request). At each step, the newly generated token projects a single Query vector $Q$ of shape `[H, 128, 1]`, which must attend against all $S$ historical Keys and Values stored in memory before passing through the layer's Feed-Forward Network (FFN).

Once prefill finishes, the prompt KV cache is **static and read-only**, while the generated output KV grows by one token per iteration (the "ragged edge"). However, in standard monolithic serving architectures, the entire historical KV cache remains permanently pinned in high-bandwidth GPU VRAM alongside the model weights.

### 1.2 Four Bottlenecks of Co-Located Serving

Coupling dynamic user KV caches with static model weights on the same accelerator creates four compounding bottlenecks:

1. **The Long-Context VRAM Capacity Wall:**
   The maximum concurrent batch size $B_{\text{max}}$ on a machine is strictly bounded by the VRAM headroom remaining after loading the layer weights:
   $$B_{\text{max}} = \left\lfloor \frac{\text{VRAM}_{\text{total}} - \text{Weights} - \text{Overhead}}{S \times \text{KV Bytes per Token}} \right\rfloor$$
   On commodity GPUs (e.g., 16 GB T4 or 24 GB L4), loading just a few layers leaves only `2–4 GB` of headroom, collapsing $B_{\text{max}}$ to $1$ or $2$ users at long contexts ($S \ge 32\text{K}$). Even on 80 GB H100s, serving 64 concurrent users at $32\text{K}$ context on a 70B model requires `640 GB` of KV cache VRAM alone—forcing operators to rent expensive frontier compute silicon merely to use its HBM as passive storage.

2. **The Private KV Memory Bandwidth Choke:**
   For a batch of $B$ concurrent users with independent conversation histories, the model weights are shared (loaded $1\times$ from VRAM), but the KV caches are private (loaded $B\times$ from VRAM):
   $$\text{Bytes Streamed per Layer} = \underbrace{W_{\text{layer}}}_{\text{Shared (Read } 1\times\text{)}} + \underbrace{B \times S \times \text{KV}_{\text{bytes}}}_{\text{Private (Read } B\times\text{)}}$$
   As $B \times S$ grows, private KV cache reads overwhelm the memory bus and throttle token generation speed.

3. **Machine Affinity ("Pegging"):**
   When a machine holds both model layer weights and User X's KV cache, User X's generation loop is **pegged** to that physical machine for hundreds of iterations. If that machine becomes overloaded, another replica cannot easily take over User X's next token without copying gigabytes of KV state across the network.

4. **The Straggler & Lockstep Batching Problem:**
   If Request 1 ($S = 512$ tokens) and Request 2 ($S = 64{,}000$ tokens) are grouped into a synchronous batch, Request 1's attention finishes $125\times$ faster than Request 2's, but Request 1 is forced to sit idle waiting for Request 2 to complete before advancing to the FFN. Conversely, if requests advance asynchronously out of lockstep on a multi-layer machine, Request 1 (wanting Layer 6) and Request 2 (wanting Layer 2) require completely different weight matrices and cannot be batched together.

---

### 1.3 Positioning: Where Our Approach Shares Ground vs. Where It Differs

To evaluate this architecture objectively, it is important to distinguish what FrontierSplit shares with recent systems research and where it diverges:

| System / Paradigm | Shared Foundation | How FrontierSplit Differs |
| :--- | :--- | :--- |
| **FlashDecoding / FlashAttention-2** *(Dao et al.)* | Uses **Online Softmax** $(A_p, m_p, \ell_p)$ to tile attention across sequence chunks in parallel and rescale without precision loss. | FlashDecoding tiles strictly inside a **single GPU's SMs**. FrontierSplit extends Online Softmax into a **Three-Tier Hierarchy** spanning Warps, SMs (with preemption/interruption intervals), and **separate network machines** over raw binary TCP. |
| **Disaggregated Prefill/Decode** *(DistServe, Splitwise, Mooncake, LMCache, vLLM)* | Recognizes that Prefill and Decode have different compute/memory profiles and separates KV storage from execution pools. | Existing systems use **Data-to-Compute**: they transfer the **entire multi-GB KV cache** over RDMA/NVLink to the Decode GPU before decoding. FrontierSplit uses **Compute-to-Data**: the KV cache stays pinned on context nodes while tiny **`8–32 KB` activation/query vectors** travel to the data. |
| **Pipeline Parallel Inference** *(Petals, Orca, FasterTransformer)* | Slices vertical transformer layers across machines and passes `[B, d_model]` activations between stages. | Standard pipelines keep the KV cache co-located with the FFN weights on each stage. FrontierSplit separates the **dynamic Attention+KV state** from **100% stateless FFN/MoE workers**, enabling unidirectional forward flow, same-layer queued GEMM batching, and per-layer micro-autoscaling. |

#### Honest Scope & Non-Goals (When NOT to Disaggregate)
Disaggregation is **not** a silver bullet for every workload. When serving a single user ($B = 1$) with short prompts ($S \le 4{,}096$) on a high-bandwidth accelerator (e.g., NVIDIA B200 or Google TPU `v4-8`) where weights and KV fit inside local HBM, network hops only add latency (`~60–80 us` per hop). A central goal of this work—and our accompanying fleet-planning tools—is providing exact mathematical boundaries showing **when to consolidate on one machine**, **when to pipeline in chunks**, and **when to fully disaggregate**.

---

## 2. Model, Layer, and KV Cache Anatomy (Unquantized BF16 Baseline)

All calculations in this work target **unquantized 16-bit precision (`BF16` / `FP16` = `2 bytes` per parameter)** to preserve full model fidelity without quantization degradation.

### 2.1 Vertical Layer Counts & Full Physical Layer Weights

A decoder-only LLM consists of $L$ vertical transformer blocks (excluding the initial token embedding table and final `lm_head` output projection). Inside each layer $i \in \{0, \dots, L-1\}$:

1. **Attention Projections:**
   * $W_q \in \mathbb{R}^{d_{\text{model}} \times (H_q \cdot d_{\text{head}})}$, $W_k \in \mathbb{R}^{d_{\text{model}} \times (H_{\text{kv}} \cdot d_{\text{head}})}$, $W_v \in \mathbb{R}^{d_{\text{model}} \times (H_{\text{kv}} \cdot d_{\text{head}})}$, and $W_o \in \mathbb{R}^{(H_q \cdot d_{\text{head}}) \times d_{\text{model}}}$.
2. **Post-Attention FFN / MoE Block (SwiGLU):**
   * In **Dense** models, a single FFN block expands $d_{\text{model}} \to d_{\text{ffn}} \to d_{\text{model}}$ using three matrices ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$):
     $$\text{Bytes}_{\text{FFN}} = 3 \times d_{\text{model}} \times d_{\text{ffn}} \times 2\text{ bytes}$$
   * In **Mixture-of-Experts (MoE)** models, the layer holds $E_{\text{total}}$ parallel expert FFNs. While only $k$ experts execute per token, the machine hosting that layer must store **all $E_{\text{total}}$ experts** in memory.

#### Table 2.1A: Open-Weight Dense Models (Unquantized BF16 Layer Sizes)

| Model | Vertical Layers ($L$) | Hidden Dim ($d_{\text{model}}$) | Heads ($H_q / H_{\text{kv}} \times d_{\text{head}}$) | FFN Vertical Shape ($d_{\text{model}} \to d_{\text{ffn}}$) | Params / Layer | **Full Layer Weight (BF16)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Meta Llama 3.1 (8B)** | 32 | 4,096 | $32 / 8 \times 128$ | $4{,}096 \to 14{,}336 \to 4{,}096$ | ~218M | **~436 MB** *(0.44 GB)* |
| **Mistral 7B** | 32 | 4,096 | $32 / 8 \times 128$ | $4{,}096 \to 14{,}336 \to 4{,}096$ | ~218M | **~436 MB** *(0.44 GB)* |
| **Google Gemma 2 (9B)** | 42 | 3,584 | $16 / 8 \times 256$ | $3{,}584 \to 14{,}336 \to 3{,}584$ | ~198M | **~396 MB** *(0.40 GB)* |
| **Alibaba Qwen 2.5 (7B)** | 28 | 3,584 | $28 / 4 \times 128$ | $3{,}584 \to 18{,}944 \to 3{,}584$ | ~233M | **~466 MB** *(0.47 GB)* |
| **Mistral NeMo (12B)** | 40 | 5,120 | $32 / 8 \times 128$ | $5{,}120 \to 14{,}336 \to 5{,}120$ | ~273M | **~545 MB** *(0.55 GB)* |
| **Qwen 2.5 (14B / 32B)** | 48 / 64 | 5,120 | $40 / 8 \times 128$ | $5{,}120 \to 27{,}648 \to 5{,}120$ | ~488M | **~975 MB** *(0.98 GB)* |
| **Google Gemma 2 (27B)** | 46 | 4,608 | $32 / 16 \times 128$ | $4{,}608 \to 36{,}864 \to 4{,}608$ | ~566M | **~1.13 GB** |
| **Meta Llama 3.1 / 3.3 (70B)** | 80 | 8,192 | $64 / 8 \times 128$ | $8{,}192 \to 28{,}672 \to 8{,}192$ | ~856M | **~1.71 GB** |
| **Alibaba Qwen 2.5 (72B)** | 80 | 8,192 | $64 / 8 \times 128$ | $8{,}192 \to 29{,}568 \to 8{,}192$ | ~878M | **~1.76 GB** |
| **Meta Llama 3.1 (405B)** | 126 | 16,384 | $128 / 8 \times 128$ | $16{,}384 \to 53{,}248 \to 16{,}384$ | ~3.19B | **~6.38 GB** |

#### Table 2.1B: Mixture-of-Experts (MoE) Models (Full Layer Size = All Experts Combined)

| Model | Vertical Layers ($L$) | $d_{\text{model}}$ | Expert Pool & Single Expert Shape ($d_{\text{model}} \to d_{\text{ffn}}$) | Single Expert Size | Params / Layer (Total) | **Full Layer Weight (BF16)** |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **Mistral Mixtral 8x7B** | 32 | 4,096 | **8 experts** ($4{,}096 \to 14{,}336 \to 4{,}096$) | ~352 MB | ~1.45B | **~2.90 GB** |
| **Mistral Mixtral 8x22B** | 56 | 6,144 | **8 experts** ($6{,}144 \to 16{,}384 \to 6{,}144$) | ~604 MB | ~2.50B | **~5.01 GB** |
| **Databricks DBRX (132B)** | 40 | 6,144 | **16 experts** ($6{,}144 \to 10{,}752 \to 6{,}144$) | ~396 MB | ~3.26B | **~6.52 GB** |
| **xAI Grok-1 (314B)** | 64 | 6,144 | **8 wide experts** ($6{,}144 \to 32{,}768 \to 6{,}144$) | ~1.21 GB | ~4.92B | **~9.84 GB** |
| **DeepSeek-V3 / R1 (671B)** | 61 *(3 dense + 58 MoE)* | 7,168 | **256 routed + 1 shared** ($7{,}168 \to 2{,}048 \to 7{,}168$) | ~88 MB | ~11.51B | **~23.01 GB** *(Dense: ~1.17 GB)* |
| **OpenAI GPT-4 Turbo** *(Est.)* | ~120 | ~12,288 | **16 experts** ($\approx 12{,}288 \to 32{,}768 \to 12{,}288$) | ~2.4 GB | ~15.0B | **~30.0 GB** |
| **Alibaba Qwen-Max** *(Est.)* | ~80–96 | ~8,192+ | **64–128 fine-grained + shared experts** | ~200–300 MB | ~25B–30B | **~50.0 – 60.0 GB** |

---

### 2.2 The Intra-Layer Weight Asymmetry: Attention vs. Post-Attention FFN

A critical architectural observation emerges when we separate the **first half of a transformer layer** (the Attention projections $W_q, W_k, W_v, W_o$) from the **second half of the layer** (the Post-Attention FFN or MoE experts):

| Model | 1st Half: Attention Weights ($W_q, W_k, W_v$ only) | 1st Half: Full Attention ($W_q, W_k, W_v, W_o$) | 2nd Half: Post-Attention FFN / Experts | **% of Layer in Post-Attention FFN** |
| :--- | :---: | :---: | :---: | :---: |
| **Llama 3.1 8B / Mistral 7B** | `48 MB` | **`80 MB`** | **`336 MB`** | **80.8%** |
| **Llama 3.1 70B** | `160 MB` | **`288 MB`** | **`1,344 MB` (`1.34 GB`)** | **82.4%** |
| **Llama 3.1 405B** | `544 MB` | **`1,056 MB` (`1.03 GB`)** | **`5,024 MB` (`4.91 GB`)** | **82.6%** |
| **Mixtral 8x7B (MoE)** | `48 MB` | **`80 MB`** | **`2,822 MB` (`2.82 GB`)** | **97.2%** |
| **DeepSeek-V3 / R1 (MoE)** | *(MLA fused)* | **`357 MB`** | **`22,643 MB` (`22.64 GB`)** | **98.5%** |

**Architectural Consequence:** The attention weights are **trivial in size (`80 MB` to `357 MB`)**, whereas **81% to 98.5%** of the static model weights reside in the post-attention FFN/MoE block, which touches **zero KV cache**!

---

### 2.3 Per-Layer and Total KV Cache Sizing Across Sequence Lengths ($S$)

Why does Llama 405B ($d_{\text{model}} = 16{,}384$) have the exact same per-layer KV cache size as Llama 8B ($d_{\text{model}} = 4{,}096$)?
Because $d_{\text{model}}$ only dictates the width of the layer weights and the vector passed *between* layers. Before saving Keys and Values into the KV cache, Grouped-Query Attention (GQA) projects the token down to $H_{\text{kv}} = 8$ heads of $d_{\text{head}} = 128$ numbers:

$$\text{KV Bytes per Token per Layer} = 2\ (\text{K \& V}) \times H_{\text{kv}} \times d_{\text{head}} \times 2\text{ bytes}$$

* **Standard Modern GQA ($H_{\text{kv}} = 8, d_{\text{head}} = 128$ — Llama 3.1 8B/70B/405B, Qwen 2.5 14B/32B/72B, Mixtral):**
  $$2 \times 8 \times 128 \times 2\text{ bytes} = \mathbf{4{,}096\text{ Bytes (4 KB) per token per layer}}$$
* **Compressed Multi-Head Latent Attention (MLA — DeepSeek-V3/R1):**
  Compresses all KV heads into a single latent vector of $512$ numbers plus $64$ RoPE positional numbers:
  $$(512 + 64) \times 2\text{ bytes} = \mathbf{1{,}152\text{ Bytes (~1.13 KB) per token per layer}}$$
* **Multi-Query Attention (MQA, $H_{\text{kv}} = 1$ — Falcon, StarCoder):**
  $$2 \times 1 \times 128 \times 2\text{ bytes} = \mathbf{512\text{ Bytes (0.5 KB) per token per layer}}$$
* *(Note: Hybrid models like Gemma 2 alternate global layers with 4K sliding-window layers; Mamba/SSM hybrids like Jamba use `0 Bytes` of KV cache on SSM layers).*

#### Table 2.3: Per-Layer vs. Full-Model KV Cache Footprint (Unquantized BF16, Single Stream $B=1$)

| Sequence Length ($S$) | **SINGLE LAYER KV Cache** <br>*(Standard GQA: 4,096 B/tok)* | **SINGLE LAYER KV Cache** <br>*(DeepSeek MLA: 1,152 B/tok)* | Full Model: **Llama 8B** <br>*(32 Layers)* | Full Model: **Llama 70B** <br>*(80 Layers)* | Full Model: **Llama 405B** <br>*(126 Layers)* | Full Model: **DeepSeek-V3** <br>*(61 Layers, MLA)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **512 tokens** | **`2.0 MB`** *(2,097,152 B)* | **`0.57 MB`** *(589,824 B)* | 64 MB | 160 MB | 252 MB | ~35 MB |
| **4,096 (4K)** | **`16.0 MB`** *(16,777,216 B)* | **`4.50 MB`** *(4,718,592 B)* | 512 MB | 1.25 GB | 1.97 GB | ~275 MB |
| **8,192 (8K)** | **`32.0 MB`** *(33,554,432 B)* | **`9.00 MB`** *(9,437,184 B)* | 1.0 GB | 2.50 GB | 3.94 GB | ~550 MB |
| **128K (131,072)** | **`512.0 MB`** *(0.50 GiB)* | **`144.0 MB`** *(0.14 GiB)* | 16.0 GB | 40.0 GB | 63.0 GB | ~8.8 GB |
| **256K (262,144)** | **`1.00 GiB`** *(1,073,741,824 B)* | **`288.0 MB`** *(0.28 GiB)* | 32.0 GB | 80.0 GB | 126.0 GB | ~17.6 GB |
| **512K (524,288)** | **`2.00 GiB`** *(2,147,483,648 B)* | **`576.0 MB`** *(0.56 GiB)* | 64.0 GB | 160.0 GB | 252.0 GB | ~35.1 GB |
| **1,024K (1M)** | **`4.00 GiB`** *(4,294,967,296 B)* | **`1.125 GiB`** *(1,207,959,552 B)* | 128.0 GB | 320.0 GB | 504.0 GB | ~68.6 GB |

> [!TIP]
> **Mental Math Rule of Thumb for a Single Layer:**
> * **Standard GQA Models:** Every **$256\text{K}$ tokens** ($262{,}144$ tokens) is **exactly $1.00\text{ GiB}$** of KV cache per layer ($1\text{M}$ tokens = $4.00\text{ GiB}$ per layer). Because $4\text{ GiB}$ easily fits on any single GPU or CPU node, **a single layer's KV cache almost never needs to be split across multiple machines for capacity**—it only needs to be split across machines if you choose to parallelize bandwidth/compute for ultra-long contexts.
> * **Compressed MLA Models:** Every **$256\text{K}$ tokens** is **exactly $288\text{ MiB}$** per layer ($\sim 3.5\times$ smaller).

---

### 2.4 Data Transferred Between Layers During Decode (`Model(i) -> Model(i+1)`)

During autoregressive token generation, **only 1 token per active request** moves from Layer $i$ to Layer $i+1$. This inter-layer activation vector has shape `[1, d_model]`, which contains the exact same number of elements as `[H_q, 128, 1]`. It is **completely independent of the prompt length $S$**:

| Model Tier | Hidden Dim ($d_{\text{model}}$) | Equivalent Head Shape `[H_q, 128, 1]` | **Single-Token Transfer Size (BF16)** | Batched Transfer Size ($B$ tokens) |
| :--- | :---: | :---: | :---: | :---: |
| **8B / Mixtral Tier** *(Llama 8B, Mixtral 8x7B)* | 4,096 | `[32, 128, 1]` | **`8,192 Bytes` (`8 KB`)** | $8\text{ KB} \times B$ |
| **DeepSeek-V3 / R1** | 7,168 | `[56, 128, 1]` | **`14,336 Bytes` (`14 KB`)** | $14\text{ KB} \times B$ |
| **70B Tier** *(Llama 70B, Qwen 72B)* | 8,192 | `[64, 128, 1]` | **`16,384 Bytes` (`16 KB`)** | $16\text{ KB} \times B$ |
| **405B Tier** *(Llama 405B)* | 16,384 | `[128, 128, 1]` | **`32,768 Bytes` (`32 KB`)** | $32\text{ KB} \times B$ |

*(Note: When transmitting partial Online Softmax states $(A_p, m_p, \ell_p)$ from a remote KV node before final normalization, keeping the accumulator $A_p$ and scalars $(m_p, \ell_p)$ in 32-bit `FP32` doubles the return payload to `16.39 KB` for a 32-head model, whereas transmitting after local normalization in `BF16` is `8.00 KB`).*

---

## 3. Datacenter Transport Tiers & Hardware Silicon Architecture

### 3.1 The 3 Datacenter Transport Tiers (Sending `8–32 KB` Packets)

At packet sizes of `8 KB` to `32 KB`, network **bandwidth** is irrelevant (`32 KB` crosses a `100 Gbps` wire in `2.5 us`). Transfer time is governed almost entirely by **hardware/software protocol latency**:

$$T_{\text{transfer}} = T_{\text{base\_latency}} + \frac{\text{Packet Bytes}}{\text{Link Bandwidth}}$$

| Transport Tier | Technology / Protocol | Where It Is Available | One-Way Transit (`8–32 KB`) | Slowdown vs. Tier 1 |
| :--- | :--- | :--- | :---: | :---: |
| **Tier 1: On-Board Hardware Bus** | **NVIDIA NVLink** / **Google TPU ICI** | Inside a single multi-GPU chassis / NVL72 rack or TPU Pod slice (not an IP network; direct copper/optical memory bus) | **`~1.0 – 2.0 us`** <br>*(0.001–0.002 ms)* | **1x (Baseline)** |
| **Tier 2: Kernel-Bypass RDMA** | **AWS EFA (SRD)** / **GCP Falcon (GPUDirect-TCPX)** / **InfiniBand** | High-end GPU instances (`p4d`, `p5`, `a3`) in the same placement group via Libfabric/NCCL | **`~12 – 25 us`** <br>*(0.012–0.025 ms)* | **~15x** |
| **Tier 3A: Raw Binary TCP (`asyncio`)** | **Persistent POSIX Socket + `TCP_NODELAY`** | **Every standard Cloud VM** (GCP, AWS, RunPod, Lambda) in the same Availability Zone | **`~60 – 80 us`** <br>*(0.060–0.080 ms)* | **~60x** |
| **Tier 3B: Standard RPC (gRPC)** | **HTTP/2 + Protobuf Serialization** | Standard Cloud VMs | **`~150 – 350 us`** <br>*(0.150–0.350 ms)* | **~250x** |

#### Why `torch.distributed` (NCCL) Does Not Fit Disaggregated Serving
While PyTorch's `torch.distributed` accesses Tier 1/2 speeds, it enforces **Single Program, Multiple Data (SPMD) gang scheduling**: every machine must launch simultaneously in a static world size, execute collective barriers in lockstep, and abort globally if any single worker stalls or scales dynamically. For elastic, asymmetric workers (where stateless FFN nodes, attention/KV nodes, and dynamic replicas operate independently), **Tier 3A persistent binary TCP (`asyncio` with `TCP_NODELAY` and zero-copy buffer framing)** provides universal cloud compatibility at `~60–80 us` per hop (`~50 us` with `uvloop` or C++ socket bindings).

---

### 3.2 The 3-Tier Silicon Hierarchy & $[N \times 128]$ Normalization

Clock frequencies across modern AI silicon vary by less than $2\times$ (`1.05 GHz` on TPU v4 to `2.04 GHz` on NVIDIA L4). Performance differences are driven by **spatial parallelism**—the physical count and layout of 16-bit Multiply-Accumulate (MAC) units firing in a single clock cycle.

To avoid conflating top-level processor clusters with internal execution engines, every accelerator can be mapped across a strict **3-Level Physical Hierarchy**:
1. **Level 1 — Compute Clusters on the Board:** Streaming Multiprocessors (**SMs**) on NVIDIA GPUs, or **TensorCores** on Google TPUs. Each cluster acts like an independent CPU core with its own local SRAM/L1 cache and its own port to the memory partitions.
2. **Level 2 — Matrix Engines (Warps / MXUs) Inside Each Cluster:** Tensor Cores (driven by warps, analogous to CPU hyperthreads/fibers sharing a core's L1 cache) or Matrix Multiply Units (**MXUs**).
3. **Level 3 — Multiplier Tile Layout Inside Each Engine:** The 2D SIMD array of 16-bit multipliers (analogous to wide 2D CPU `XMM`/`AVX` vector registers).

Because transformer attention heads universally standardize on **$d_{\text{head}} = 128$**, we normalize every chip's total multiplier count by **128**:

$$\text{Normalized Board Capacity} = N \times 128, \qquad \text{where } N = \frac{\text{Total Physical 16-bit MACs}}{128}$$

Here, **$N$** is the exact number of **128-element vectors** the board can multiply simultaneously in a single clock tick:

#### Table 3.2: Physical Silicon Hierarchy & $[N \times 128]$ Normalization

| Accelerator | **Level 1: Compute Clusters on Board** | **Level 2: Engines per Cluster** | **Level 3: Multipliers per Engine (Native Tile)** | **Total Board Multipliers ($M_{\text{total}}$)** | **Normalized Capacity ($N \times 128$)** | Tokens/Cycle Across $H=32$ Heads ($N / 32$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA T4** *(16 GB)* | **40 SMs** | 8 Tensor Cores/SM | 64 MACs ($16 \times 16$ tile) | **20,480** | **$160 \times 128$** | **5.0 tokens/cycle** |
| **NVIDIA L4** *(24 GB)* | **58 SMs** | 4 Tensor Cores/SM | 128 MACs ($16 \times 16$ tile) | **29,696** | **$232 \times 128$** | **7.25 tokens/cycle** |
| **Google TPU v4** *(1 MXU)* | *(Sub-unit)* | 1 MXU | 16,384 MACs ($128 \times 128$ grid) | **16,384** | **$128 \times 128$** | **4.0 tokens/cycle** |
| **Google TPU v4** *(1 Chip, 32 GB)* | **2 TensorCores** | 2 MXUs/TensorCore | 16,384 MACs ($128 \times 128$ grid) | **65,536** | **$512 \times 128$** | **16.0 tokens/cycle** |
| **Google TPU `v4-8`** *(1 Host, 4 Chips, 128 GB)* | **8 TensorCores** | 2 MXUs/TensorCore | 16,384 MACs ($128 \times 128$ grid) | **262,144** | **$2{,}048 \times 128$** | **64.0 tokens/cycle** |
| **NVIDIA A100** *(80 GB)* | **108 SMs** | 4 Tensor Cores/SM | 256 MACs ($16 \times 8$ tile) | **110,592** | **$864 \times 128$** | **27.0 tokens/cycle** |
| **NVIDIA H100** *(80 GB)* | **132 SMs** | 4 Tensor Cores/SM | 512 MACs ($64 \times 64$ tile) | **270,336** | **$2{,}112 \times 128$** | **66.0 tokens/cycle** |
| **Google TPU `v5p-8`** *(1 Host, 4 Chips, 380 GB)* | **8 TensorCores** | 4 MXUs/TensorCore | 16,384 MACs ($128 \times 128$ grid) | **524,288** | **$4{,}096 \times 128$** | **128.0 tokens/cycle** |
| **NVIDIA B200** *(192 GB)* | **160 SMs** | 4 Tensor Cores/SM | 1,024 MACs ($64 \times 128$ tile) | **~625,000** | **$\approx 4{,}880 \times 128$** | **~152.5 tokens/cycle** |

*(Note on Google Cloud TPU `-8` Notation: In Google Cloud terminology, the suffix number denotes total **TensorCores** on the host, where each TPU v4/v5p chip contains 2 TensorCores. Thus, a single `v4-8` host VM contains **4 physical TPU v4 chips** = 8 TensorCores = 16 MXUs = `128 GB` HBM).*

#### Two Natural Hardware Tiling Modes for `[H, 128, S]`
Given data formatted as `[Heads, 128, Tokens]` (e.g., `[32, 128, S]`) and a $128 \times 128$ hardware array (`16,384` multipliers):
* **Mode 1 — All Heads Packed (`[32, 128, 4]`):** Load $Q$ for all 32 heads (`[32, 128, 1]`) into registers and process **4 tokens of history across all 32 heads simultaneously** ($32 \times 128 \times 4 = 16{,}384$ multipliers = $100\%$ utilization).
* **Mode 2 — Deep Token Tile per Head (`[1, 128, 128]`):** Load $Q$ for 1 head (`[1, 128, 1]`) into registers and process **128 contiguous tokens of history** for that head in one wave ($1 \times 128 \times 128 = 16{,}384$ multipliers = $100\%$ utilization).

---

### 3.3 The $C$-Cycle VRAM-to-SRAM Window, Sliding-Window Warps, and Memory Segments

While the normalized multiplier formula gives pure compute cycles:

$$\text{Compute Cycles} \propto \frac{B \times S \times H}{N}$$

actual execution time on an accelerator is governed by the interaction between **VRAM-to-SRAM bandwidth**, **Warp/SM scheduling**, and **Memory Partitions (Segments)**.

#### 1. The $C$-Cycle Memory Refill Window
A warp or MXU cannot execute until its full operand tile has been staged from off-chip VRAM (GDDR6/HBM) into on-chip SRAM/registers. Across all modern accelerators, the memory bus delivers roughly `80` to `286 Bytes per clock cycle` per $128 \times 128$ multiplier group. Loading a single $128 \times 128$ BF16 tile (`32,768 Bytes`) from VRAM into SRAM therefore requires **$C$ clock cycles**:

| Accelerator | Clock Speed | VRAM Bandwidth | Bytes Pulled / Cycle *(per $128 \times 128$ group)* | **$C$ (Cycles to Load One `32 KB` Tile from VRAM $\to$ SRAM)** |
| :--- | :---: | :---: | :---: | :---: |
| **Google TPU v4** | 1.05 GHz | 1,200 GB/s | ~286 B/cycle | **$C \approx 115\text{ cycles}$** |
| **NVIDIA A100** | 1.41 GHz | 2,039 GB/s | ~241 B/cycle | **$C \approx 136\text{ cycles}$** |
| **NVIDIA T4** | 1.59 GHz | 300 GB/s | ~151 B/cycle | **$C \approx 217\text{ cycles}$** |
| **NVIDIA B200** | 1.80 GHz | 8,000 GB/s | ~117 B/cycle | **$C \approx 280\text{ cycles}$** |
| **NVIDIA H100** | 1.83 GHz | 3,350 GB/s | ~110 B/cycle | **$C \approx 298\text{ cycles}$** |
| **NVIDIA L4** | 2.04 GHz | 300 GB/s | ~81 B/cycle | **$C \approx 405\text{ cycles}$** |

#### 2. The Sliding-Window Firehose Across Warps & The $\lfloor C / k \rfloor$ Reuse Rule
Inside an SM, the memory controller does not trickle a few bytes to every warp simultaneously. Instead, it acts as a **sliding-window firehose**: it fills Warp 0's SRAM tile first, and as soon as Warp 0 is full, Warp 0 executes its matrix operation (taking $k$ cycles) while the firehose slides to fill Warp 1, then Warp 2, and so on.

```
VRAM Firehose (Sliding Window Across Warps):
[Fill W0] ──► [Fill W1] ──► [Fill W2] ──► ... ──► [Fill W_last] ──► [Return to W0]
│                                                                     ▲
└────── Warp 0 runs tile op (k cycles), then waits (C - k) cycles ────┘
```

Because the firehose takes **$C$ total cycles** to cycle through the other warps and return to Warp 0 with its *next* tile from VRAM:
* **If $B = 1$ (No Data Reuse):** Warp 0 finishes its $k$-cycle operation and sits **idle for $C - k$ cycles** waiting for the next tile to arrive from VRAM.
* **If $B > 1$ Sharing the Loaded Tile (CPU Fibers Sharing L1 Cache):** Because Warp 0 *still holds the current tile in its SRAM/registers* during those $C - k$ waiting cycles, it can execute up to **$\left\lfloor \frac{C}{k} \right\rfloor$ operations** for other queued messages against that exact same loaded tile **for free (zero extra wall-clock time)** before the firehose returns with the next tile!

#### 3. Parallel Memory Segments (Why Single-SM Work Cannot Saturate Bandwidth)
Crucially, a GPU's `300 GB/s` or `3,350 GB/s` memory bus is **not** a single pipe connected to one warp or one SM. Physical VRAM is divided across multiple independent **Memory Partitions / HBM Channels** wired across the chip's SMs/GPCs (and on a TPU v4, split across the 2 independent TensorCores at `600 GB/s` each).
* A single warp or a small handful of SMs can only pull from its local memory segment—leaving the remaining memory segments idle.
* To reach the board's peak memory bandwidth, work **must** be distributed across all SMs/TensorCores simultaneously—either by spreading a single large message across all SMs, or by **striping multiple concurrent messages across different SMs/TensorCores**.

---

## 4. Disaggregated Pipeline Topologies & Three-Tier Chunking

### 4.1 The Multi-Layer "Ping-Pong" Collision Problem

Suppose a single model machine holds multiple layers (e.g., Layers $0$ through $7$) while the KV caches live on remote KV machines. When we try to keep the model machine busy by interleaving two messages (Message 1 and Message 2), two problems occur:

```
TIME (ms)   MODEL MACHINE (Holds Layers 0–7)          REMOTE KV(0) MACHINE
─────────   ────────────────────────────────          ────────────────────
0.00 ms     Runs Message 1 (Layer 0 Q-proj)
0.05 ms     Sends Q1 to KV(0) ──────────────────────► Computes KV(0) for Msg 1...
            [Model Machine is now IDLE]               (Takes ~0.20 ms)
0.05 ms     Grabs Message 2 from queue!
            Starts Message 2 (Layer 0 / Layer 3)...
0.25 ms     [COLLISION / QUEUE STALL!]                ◄── Message 1 returns from KV(0)!
            Message 1 is ready to finish Layer 0,
            but Model Machine is busy with Message 2!
            [Message 1 must wait or contend]
```

1. **The $2\times$ Network Ping-Pong Tax:** Across 8 layers on the same machine, the token must go out to `KV(i)` and come back to the model machine **8 times in a row** ($8 \times 2 = 16$ network hops = `~1.12 ms` of pure network transit per token).
2. **Inter-Layer Queue Collisions:** When Message 1 returns from `KV(0)` ready to run Layer 0's FFN, the machine may be mid-execution on Layer 3 for Message 2. Because Layer 0 and Layer 3 use different weight matrices, they **cannot be batched together**!

---

### 4.2 Resolving the Intra-Layer Dataflow: Three Pipeline Topologies

To eliminate multi-layer collisions and turn the system into a smooth pipeline where each machine is dedicated to a single layer stage, we must account for everything that happens inside Layer $i$:

$$\text{Input } x_i \xrightarrow{W_{q,k,v}} (Q_i, K_i, V_i) \xrightarrow{\text{Attention with KV}_i} \text{Attn}_i \xrightarrow{W_o\ +\ \text{Residual } x_i} h_i \xrightarrow{\text{FFN}_i\ +\ \text{Residual } h_i} x_{i+1}$$

Notice that after `KV(i)` computes attention, Layer $i$ still requires:
1. Merging any partial Online Softmax states $(A_p, m_p, \ell_p)$,
2. Multiplying by $W_{o,i}$ and **adding the original input residual $x_i$** (`8–32 KB`), and
3. Running **$\text{FFN}_i$** (`81%–98.5%` of the layer's weights) to produce $x_{i+1}$.

This yields **three valid architectural topologies** for wiring Layer $i$:

```
OPTION A: Out-and-Back (Standard Full-Layer Worker + Weightless KV Node)
┌───────────────────────────┐   1. Send Q_i, K_i, V_i   ┌───────────────────────────┐
│      Model Worker (i)     │ ────────────────────────► │        KV Node (i)        │
│ Holds: W_qkv, W_o, FFN_i  │ ◄──────────────────────── │ Holds: 0 Weights, KV_i    │
└─────────────┬─────────────┘   2. Return (A_p, m_p, l) └───────────────────────────┘
              │ 3. Finish W_o + x_i + FFN_i, send x_{i+1} (8–32 KB)
              ▼
┌───────────────────────────┐
│     Model Worker (i+1)    │
└───────────────────────────┘

OPTION B: Half-Layer Staggered Conveyor Belt (Forward-Only, Weightless KV Node)
┌───────────────────────────┐   1. Send Q_i, K_i, V_i   ┌───────────────────────────┐
│     Stage Worker (i)      │ ────────────────────────► │        KV Node (i)        │
│ Holds: FFN_{i-1} + W_qkv_i│ ────┐                     │ Holds: 0 Weights, KV_i    │
└───────────────────────────┘     │ Pass residual x_i   └─────────────┬─────────────┘
                                  ▼                                   │ 2. Forward (A_p, m_p, l)
                            ┌─────────────────────────────────────────▼─────────────┐
                            │                   Stage Worker (i+1)                  │
                            │       Holds: W_o_i + FFN_i  AND  W_qkv_{i+1}          │
                            └───────────────────────────────────────────────────────┘

OPTION C: Co-Located Attention+KV Node ──► Stateless FFN Node (Pure Forward Split)
┌───────────────────────────────────────────────────────┐
│            Attention + KV Node (i)                    │
│  • Static Weights: W_q, W_k, W_v, W_o (Only 80–357 MB)│
│  • Dynamic State:  Layer i KV Cache (1 GB / 256K tok) │
│  1. Receives x_i (8–32 KB)                            │
│  2. Computes Q_i, K_i, V_i locally                    │
│  3. Appends new K_i, V_i directly to local KV cache!  │
│  4. Runs Attention + SoftMax + W_o + adds residual x_i│
└───────────────────────────┬───────────────────────────┘
                            │ 1 Hop: Sends h_i (8–32 KB, strictly forward!)
                            ▼
┌───────────────────────────────────────────────────────┐
│            Stateless FFN / Expert Node (i)            │
│  • Static Weights: FFN_i / MoE Experts (81%–98.5%)    │
│  • Dynamic State:  ZERO KV Cache (100% Stateless!)    │
│  1. Runs heavy FFN_i + adds residual h_i              │
│  2. Sends x_{i+1} (8–32 KB) forward to Attn+KV (i+1)  │
└───────────────────────────────────────────────────────┘
```

#### Comparison of the Three Topologies

| Feature | **Option A: Out-and-Back** | **Option B: Half-Layer Shift** | **Option C: Co-Located Attn+KV $\to$ FFN** |
| :--- | :--- | :--- | :--- |
| **Network Hops per Layer** | **3 hops** (`Model(i) -> KV(i) -> Model(i) -> Model(i+1)`) | **2 hops** (`Stage(i) -> KV(i) -> Stage(i+1)`) | **2 hops** (`Attn_KV(i) -> FFN(i) -> Attn_KV(i+1)`) |
| **Direction of Data Flow** | Bi-directional (ping-pong at each layer) | Strictly forward (unidirectional) | Strictly forward (unidirectional) |
| **Residual $x_i$ Handling** | Stays in `Model(i)` memory | Must be sent (`8–32 KB`) from `Stage(i)` to `Stage(i+1)` | Added locally on `Attn_KV(i)`; zero extra network data |
| **New Token $K_i, V_i$ Storage** | Shipped over network to `KV(i)` | Shipped over network to `KV(i)` | **Generated & chunked locally in memory on `Attn_KV(i)`** |
| **Weights on KV Machine** | `0 MB` (Weightless) | `0 MB` (Weightless) | **`80–357 MB`** ($W_{q,k,v,o}$ — tiny fraction of layer) |
| **Heavy FFN/MoE Machine** | Coupled with $W_{q,k,v,o}$ and interrupted twice | Coupled with next layer's $W_{q,k,v}$ | **100% isolated, 100% stateless, runs 1 uninterrupted pass** |

> [!IMPORTANT]
> **Why Option C Is Special:** Because the Attention weights ($W_q, W_k, W_v, W_o$) are only `80 MB` (Llama 8B / Mixtral) to `288 MB` (Llama 70B), placing them directly on the KV cache machine costs almost zero VRAM, eliminates shipping new $K, V$ vectors or residuals over the network, makes chopping/committing newly generated KV chunks a local memory operation, and leaves the heavy FFN/MoE machines (`1.34 GB` to `22.6 GB` per layer) completely stateless!

---

### 4.3 The Three-Tier Chunking Hierarchy & Dynamic Striping ("Interruption Intervals")

How should work be divided inside and across machines so that a single request achieves maximum speed when the machine is empty, while multiple concurrent requests can share the hardware without head-of-line blocking?

We structure execution into a **Three-Tier Chunking Hierarchy** aligned directly with the silicon hierarchy from Section 3.2:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 3: MULTI-MACHINE / NETWORK CHUNKING (Online Softmax Across Nodes)                 │
│   • Used only when S exceeds single-node capacity/bandwidth target (e.g., S > 256K–1M) │
│   • Merges partial (A_p, m_p, ℓ_p) across nodes via exact FP32 Online Softmax          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 2: SM / TENSORCORE CHUNKING (The "Interruption Interval" & Striping Quantum)      │
│   • Divides sequence S (or batch B) into chunks assigned to SMs / TensorCores          │
│   • Empty Machine: Spreads chunks across 100% of SMs to activate all memory segments   │
│   • Multi-Tenant Load: Stripes SMs across concurrent messages; yields SMs at chunk     │
│     boundaries ("Interruption Intervals") since an active chunk cannot be preempted    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 1: WARP / MXU TILE CHUNKING (Register-Level Execution)                            │
│   • Fixed to native hardware tile (16×16 Tensor Core warp tile or 128×128 MXU wave)    │
│   • Prologue loads Q into warp registers once; streams token tiles through SRAM        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 1. Tier 1: Warp / Tile Chunking (And the Register Load Prologue Tax)
At the lowest level, tokens are grouped into the native tile of a Warp or MXU (`16` to `128` tokens).
* When a warp begins executing a chunk, it pays a **prologue tax**: loading its slice of the Query vector $Q$ (`[1, 128]`) into local registers and initializing its Online Softmax accumulators $(A_p, m_p, \ell_p)$.
* While $Q$ stays pinned in registers, the warp streams consecutive $K, V$ tiles through SRAM, and finally pays an **epilogue tax** to write $(A_p, m_p, \ell_p)$ back to shared memory.

#### 2. Tier 2: SM / TensorCore Chunking & The "Interruption Interval" Trade-Off
Once a chunk of work is launched onto an SM or TensorCore, **the hardware cannot preempt or interrupt it mid-chunk**. This creates a fundamental design knob for multi-tenant scheduling:
* **When the machine has only 1 message:** We divide the message's KV cache across **all SMs / TensorCores** so every memory segment pulls in parallel at maximum bandwidth.
* **When a new message arrives while a long KV cache is running:** If the first message launched all $128\text{K}$ tokens as one monolithic, uninterruptible job across every SM, the newly arrived message would be blocked until the entire $128\text{K}$ context finished.
* **The "Interruption Interval" Solution:** By sizing each **Tier 2 SM/TensorCore chunk** to a moderate token count (e.g., $1{,}024$ to $4{,}096$ tokens per SM chunk), completion of each SM chunk creates a natural **interruption boundary (preemption quantum)**. At that boundary, the scheduler can dynamically **re-stripe the SMs**—allowing newly arrived messages to claim a subset of SMs/TensorCores while the remaining chunks of the long message continue on the rest!
* **Balancing the Knob (Chunk Size vs. Register Prologue Overhead):**
  * If Tier 2 SM chunks are **too large**, interruption intervals are too far apart, increasing queuing jitter for newly arriving messages.
  * If Tier 2 SM chunks are **too small**, each warp repeatedly pays the **Tier 1 register-load prologue and Online Softmax merge epilogue tax** at every boundary, wasting compute cycles.
* **The Small-Context Edge Case ($S$ Too Small to Fill All SMs):**
  If a message has a short KV cache (e.g., $S = 512$ or $1{,}024$ tokens), spreading it across all 108 SMs would give each SM only a few tokens—wasting most of the time on register-load prologues. Instead, a short KV cache is packed onto **only a few SMs (e.g., 4–8 SMs)**, leaving the other 100 SMs and their memory segments immediately open for other concurrent messages!

#### 3. Same-Layer Model Batching ($B \gg 1$) vs. Serial KV Cache Usage ($B = 1$)
A fundamental traffic asymmetry governs live production clusters:
1. **Model Weights Are Shared *Simultaneously* Across Concurrent Streams ($B \gg 1$):**
   At any millisecond, all active user streams in the cluster require the **exact same static layer weights**. On a dedicated single-layer FFN worker (or during the shared $W_{q,k,v,o}$ weight projections), any queued requests automatically stack into a single shared-weight matrix multiply (`[B, d_model] x W_layer`), reading the weight from VRAM **once** and reusing each loaded SRAM tile up to $\lfloor C / k \rfloor$ times during the $C$-cycle memory window.
2. **KV Caches Are Used *Serially* Over Time ($B = 1$ per Active KV Cache):**
   Unlike model weights, different users do not share the same conversation KV cache, and even a single user interacts with their KV cache **serially** (Turn 1 completes before Turn 2 begins). Consequently, at any given instant in the cluster, **almost every active KV cache is queried by only a single stream ($B = 1$)**!
   * Because a KV cache almost never sees $B > 1$ simultaneous queries against the same historical tokens (outside of niche parallel tree-of-thought branching), **KV cache attention cannot rely on batching $B > 1$ to fill the $C$-cycle multiplier window**—it is almost always a $B = 1$ memory-bandwidth-bound operation per cache.
   * Instead, when multiple active streams hit the same KV machine simultaneously, they are accessing *different* KV caches. They share the machine via **Tier 2 Spatial Striping across SMs/TensorCores** (activating parallel memory segments), rather than $B > 1$ matrix batching!
3. **Demand-Based Micro-Autoscaling:** If a queue builds up at a specific bottleneck stage, the cluster can spin up an extra replica of **just that single layer stage** (requiring only `~2–4 GB` of VRAM) rather than duplicating an entire `140+ GB` model!

---

## 5. Analytical Hardware Performance & Cost Modeling

### 5.1 Step-by-Step Single-Stream Roofline Walkthrough (Llama 3.1 70B, Unquantized BF16)

To establish the baseline for a single user request in isolation ($B = 1$, $S = 4{,}096$ tokens, 80 layers, `1.71 GB` weights + `0.016 GB` KV = `1.726 GB` per layer), we compare pure compute time against memory-bus streaming time and network hop latency across each accelerator family:

$$\text{Time per Layer} = \max\Big(T_{\text{MAC\_compute}},\ T_{\text{VRAM\_stream}}\Big), \qquad \text{TPS} = \frac{1}{(80 \times T_{\text{layer}}) + (\text{Cuts} \times T_{\text{net}})}$$

| Accelerator | Setup Required to Hold 80 Layers (`~138 GB`) | Aggregate Memory Bus | Pure MAC Compute Time *(All 80 Layers)* | VRAM Streaming Time *(All 80 Layers)* | Network Transit Penalty | **Total Time / Token** | **Single-User Speed (TPS)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA T4** *(16 GB)* | **10 GPUs** *(8 layers/GPU, TCP)* | 300 GB/s *(per GPU)* | 2.10 ms | **460.0 ms** | +0.68 ms *(9 TCP cuts)* | **460.7 ms** | **~2.2 tok/s** |
| **NVIDIA L4** *(24 GB)* | **7 GPUs** *(12 layers/GPU, TCP)* | 300 GB/s *(per GPU)* | 1.13 ms | **460.0 ms** | +0.45 ms *(6 TCP cuts)* | **460.5 ms** | **~2.2 tok/s** |
| **Google TPU v4** *(1 Chip, 32 GB)* | **5 Chips** *(16 layers/chip, ICI)* | 1,200 GB/s *(per chip)* | 0.99 ms | **115.0 ms** | +0.008 ms *(4 ICI cuts)* | **115.0 ms** | **~8.7 tok/s** |
| **NVIDIA A100** *(80 GB)* | **2 GPUs** *(40 layers/GPU, NVLink)* | 2,039 GB/s *(per GPU)* | 0.44 ms | **67.7 ms** | +0.001 ms *(1 NVLink)* | **67.7 ms** | **~14.8 tok/s** |
| **NVIDIA H100** *(80 GB)* | **2 GPUs** *(40 layers/GPU, NVLink)* | 3,350 GB/s *(per GPU)* | 0.14 ms | **41.2 ms** | +0.001 ms *(1 NVLink)* | **41.2 ms** | **~24.3 tok/s** |
| **Google TPU `v4-8`** *(4 Chips, 128 GB + KV offload)* | **1 Host** *(Tensor/Pipeline across 4 chips)* | 4,800 GB/s *(aggregate)* | 0.25 ms | **28.8 ms** | +0.006 ms *(ICI)* | **28.8 ms** | **~34.7 tok/s** |
| **NVIDIA B200** *(192 GB)* | **1 GPU (Solo)** *(All 80 layers fit!)* | 8,000 GB/s | 0.06 ms | **17.3 ms** | **0.000 ms** *(0 cuts)* | **17.3 ms** | **~57.9 tok/s** |

---

### 5.2 Cloud Economics: Hourly Rental & Cost per 1 Million Tokens ($B=1$ vs. Batched)

| Accelerator | On-Demand Price *(per Chip/GPU)* | Spot Price *(per Chip/GPU)* | Cluster Spot Cost *(Llama 70B)* | Single-User ($B=1$) Cost / 1M Tokens | **Why This Happens (Economic Insight)** |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **NVIDIA T4** *(10x Fleet)* | ~$0.55 / hr | ~$0.15 / hr | **~$1.50 / hr** | **~$189.00** | **The "Cheap GPU" Latency Trap:** At $B=1$, 300 GB/s GDDR6 takes 460 ms/token. Only cost-effective when $B \ge 32$ via same-layer batching! |
| **NVIDIA L4** *(7x Fleet)* | ~$0.84 / hr | ~$0.28 / hr | **~$1.96 / hr** | **~$247.00** | Same 300 GB/s memory bus as T4; requires high batch concurrency ($B \ge 32$) to amortize weight reads. |
| **NVIDIA A100** *(2x 80GB)* | ~$3.67 / hr | ~$1.35 / hr | **~$2.70 / hr** | **~$50.60** | 2 TB/s HBM2e cuts token time by $6.8\times$ vs T4. |
| **NVIDIA H100** *(2x 80GB)* | ~$3.50–$4.50 / hr | ~$2.25 / hr | **~$4.50 / hr** | **~$51.40** | Faster per token (`24.3 tok/s`), similar cost/token at $B=1$; pulls ahead rapidly at $B \ge 16$. |
| **Google TPU `v4-8`** *(4 Chips)* | $12.88 / hr | ~$4.50 / hr | **~$4.50 / hr** | **~$36.00** | High aggregate HBM bandwidth over optical ICI at low spot cost. |
| **NVIDIA B200** *(1x 192GB)* | ~$5.50–$6.50 / hr | ~$6.00 / hr | **~$6.00 / hr** | **~$28.70** | Entire 70B model fits on 1 chip with 8 TB/s HBM3e and zero network hops. |

---

## 6. Fleet Sizing Methodology & Practitioner Guidelines

To help engineers choose the right architecture based on adjustable workload knobs (**Model Size**, **Prompt Length $S$**, **Concurrent Users $B$**, and **Optimization Priority**), we establish the following decision matrix:

### 6.1 The Fleet Decision Matrix

| Scenario / Knobs | Primary Bottleneck | Recommended Architecture | Recommended Hardware |
| :--- | :--- | :--- | :--- |
| **1. Low Latency, Short Context** <br>($B = 1\text{–}4$, $S \le 8\text{K}$) | Sequential memory streaming speed & network hop latency | **Consolidated Single-Box (Do NOT Split Over TCP)** | **1x B200**, **1x H200**, **2x H100 (NVLink)**, or **TPU `v4-8` / `v5p-8`** |
| **2. High Concurrency, Long Context** <br>($B \ge 16$, $S = 32\text{K–}256\text{K}$) | KV Cache VRAM capacity wall & multi-user straggler stalls | **FrontierSplit Option C (Co-Located Attn+KV $\to$ Stateless FFN Pipeline)** | **Stateless H100/A100/L4 FFN Nodes** + **High-Capacity `Attn+KV` Nodes** |
| **3. Massive Context ($S \ge 256\text{K}$)** <br>(Serial per-user contexts, $B=1$ per KV) | Single-layer KV bandwidth ($B=1$) & multi-user SM contention | **Option C + Tier 2 SM Striping & Tier 3 Multi-Node Online Softmax Fan-Out** | **Memory-Bandwidth `Attn+KV` Nodes** (3-tier chunking) + **Batched ($B \gg 1$) Stateless FFN Nodes** |
| **4. Budget Batch Processing** <br>($B \ge 64$, latency-insensitive) | Dollars per million tokens ($\text{TPS} / \text{\$/hr}$) | **Unidirectional Pipeline with Large Same-Layer FFN Batching ($B \approx C/k$)** | **Google TPU `v5e`** or **L4/T4 Fleet** saturating the $C$-cycle memory window on model weights |

### 6.2 Three Red Flags (What NOT to Do)

1. **Red Flag 1: Using T4/L4 Fleets for Single-User ($B=1$) Interactive Chat:**
   Because a T4/L4 only has `300 GB/s` of VRAM bandwidth, running $B=1$ wastes $99\%$ of the $C$-cycle memory window (`217` to `405` cycles per tile load), resulting in `~2 tok/s` and costing **more per token** than an H100 or B200. T4/L4 fleets should only be used for model/FFN stages when multi-user queues ($B \ge 16\text{–}64$) reuse the loaded SRAM weight tiles during those $C$ cycles.
2. **Red Flag 2: Renting 80 GB H100s to Hold Static Long-Context KV Caches:**
   For Mixtral 8x7B or DeepSeek-V3, $97\%$ to $98.5\%$ of the layer is static FFN/MoE weights that need zero KV cache. Piling hundreds of gigabytes of static user KV caches onto the same GPUs hosting the FFN weights chokes $B_{\text{max}}$ and wastes frontier compute silicon.
3. **Red Flag 3: Multi-Layer TCP Ping-Pong Without Intra-Layer Decoupling:**
   Hosting 8–10 layers on a single worker and making synchronous out-and-back TCP calls at every layer incurs $2\times$ network hops per layer and causes returning KV packets to collide with in-flight layers of other messages. Either group layers locally with co-located KV, or decouple into **Option C (`Attn+KV(i) -> FFN(i)`)** unidirectional stages with **Tier 2 SM interruption intervals**.
4. **Guideline Note (Spatial Layer-to-SM/TensorCore Ceiling on Large Machines):**
   When spatially striping multiple layers to run concurrently on a single large machine, you cannot run more layers simultaneously in parallel than the number of independent compute clusters (**SMs / TensorCores**—or minimum SM groups needed per memory segment) on that board ($K_{\text{layers}} \le N_{\text{clusters}}$). On TPUs with few giant clusters (e.g., `2 TensorCores` per chip or `8 TensorCores` on a `v4-8`), or on large GPUs where each active layer stripe requires a minimum group of SMs, exceeding this ratio forces layers to time-slice and queue rather than stream in parallel. *(Saved for deeper exploration in the guidelines phase).*

---

# TBD
