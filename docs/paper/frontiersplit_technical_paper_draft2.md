# FrontierSplit: Hardware-Aligned Disaggregated Attention, Unidirectional Pipeline Streaming, and Fleet Optimization for Long-Context LLMs

**Authors:** FrontierSplit Research Team  
**Repository:** [github.com/frontiersplit/frontiersplit](https://github.com/frontiersplit/frontiersplit)  
**Status:** Technical Specification & Working Paper Draft (v2.2 — Pass 1b Primary-Source & Technical Corrections)

---

## Abstract

Serving unquantized Large Language Models (LLMs) at long context lengths ($S = 32\text{K to } 1\text{M+}$ tokens) exposes a structural conflict between **static model weights** and **dynamic Key-Value (KV) caches**. Across open-weight dense and Mixture-of-Experts (MoE) architectures, static layer weights are shared across active user requests and concentrated ($77.8\%$ to $98.4\%$ of resident bytes) in post-attention Feed-Forward Networks (FFNs) or expert pools. By contrast, each conversation's KV cache is private to that session, grows linearly with sequence length ($1.00\text{ GiB}$ per $256\text{K}$ tokens per layer in standard GQA), and—even when an attention kernel batches single-token queries from multiple sessions—must generally be read separately per session without cross-request memory reuse. Co-locating both on monolithic GPU clusters creates a **Long-Context VRAM Capacity Bottleneck**, **Session-to-Machine Affinity ("Pegging")**, and **Mixed-Length Scheduling Contention**.

We present **FrontierSplit**, an analytical performance-modeling framework and disaggregated inference architecture built around three components:
1. **Compute-to-Data Disaggregation via Online Softmax:** Rather than migrating multi-gigabyte KV caches across the network (Data-to-Compute), workers transmit compact `8–32 KiB` single-token activation or query vectors (`~60–80 us` one-way transit over persistent binary `asyncio` TCP) and combine sequence-partitioned KV chunks via an algebraically exact **Online Softmax decomposition** evaluated in FP32 arithmetic.
2. **Intra-Layer Partitioning & Unidirectional Streaming:** We compare three distributed intra-layer topologies—*Out-and-Back*, *Half-Layer Staggered*, and *Co-Located Attention+KV $\to$ KV-Stateless FFN*. Because attention projections ($W_q, W_k, W_v, W_o$) account for $1.6\%$ to $22.2\%$ of resident layer weights (`56–1,088 MiB` across `7B` to `405B` models), co-locating them on the KV node keeps $K, V$ projection and chunk management local while forwarding a single `8–32 KiB` hidden state per token to KV-stateless FFN/MoE workers.
3. **Sequence-Chunked KV Striping & Roofline-Guided Fleet Sizing:** Using the idealized weight-only Roofline crossover ($B_{\text{crit}} \approx F_{\text{peak}} / \text{BW}_{\text{peak}}$) alongside a normalized $[N \times 128]$ MAC capacity notation, we describe a **three-level sequence-chunking scheme** (warp/MXU tile $\to$ tunable SM/TensorCore interruption interval $\to$ multi-node Online Softmax) supporting intra-request *self-striping* and inter-request *concurrency-striping*, and formulate testable operating hypotheses across NVIDIA GPUs (T4, L4, A100, H100, HGX B200) and Google Cloud TPUs (v4, v5p).

---

## 1. Introduction & Motivation

### 1.1 The Asymmetric Reality of Transformer Inference

Autoregressive generation in decoder-only transformers consists of two distinct operational phases:
* **Prefill (Prompt Ingestion):** Processes $S$ input tokens in parallel, computing initial self-attention and populating the Key-Value (KV) cache across all $L$ layers.
* **Decode (Token Generation):** Generates output tokens sequentially ($1$ token per step per session). At each decode step, the newly generated token projects a single Query tensor $Q$ of shape `[H_q, d_head, 1]` (commonly $d_{\text{head}} = 128$), attends over all $S$ historical Keys and Values stored for that session, and passes through the layer's Feed-Forward Network (FFN) or routed MoE experts.

Once prefill finishes, committed prompt KV blocks remain read-only on fixed chunk boundaries, while newly generated tokens append one token per iteration to an active partial tail chunk (the **ragged edge**), which is migrated into the aligned KV store once a full chunk boundary is reached. In conventional monolithic serving, however, the entire historical KV footprint remains pinned in accelerator HBM alongside static model weights.

### 1.2 Four Bottlenecks of Co-Located Serving

Coupling dynamic per-session KV caches with static model weights on the same accelerator introduces four compounding constraints:

1. **The Long-Context VRAM Capacity Wall:**
   The number of concurrent active sessions $M_{\text{node}}$ that can co-reside on a machine is bounded by the software-usable VRAM remaining after loading resident model weights and runtime workspace:
   $$M_{\text{node}} = \left\lfloor \frac{\text{VRAM}_{\text{usable}} - \text{Weights}_{\text{resident}} - \text{Workspace}}{S \times \text{KV Bytes per Token}} \right\rfloor$$
   On older or lower-capacity GPUs (such as `16 GiB` physical / `~15 GiB` usable NVIDIA T4 or `24 GiB` physical / `~22.5 GiB` usable NVIDIA L4), hosting several unquantized layers leaves only `2–4 GiB` of headroom, limiting long-context capacity ($S \ge 32\text{K}$) to one or two sessions. Even on `80 GiB` H100s, at $S = 32{,}768$ tokens on an 80-layer GQA model (`Llama 3.1 70B`, $4{,}096\text{ B/tok/layer}$), each session's KV cache occupies $32{,}768 \times 4{,}096 \times 80 = \mathbf{10.00\text{ GiB}}$; serving $64$ concurrent sessions therefore requires $64 \times 10.00\text{ GiB} = \mathbf{640.0\text{ GiB}}$ of KV state alone—the entire HBM capacity of more than eight `80 GB` H100 GPUs.

2. **Private KV Memory Bandwidth Contention:**
   When $B$ concurrent requests execute a dense layer together, resident layer weights are read once from VRAM and reused across the batch, whereas private KV caches must generally be read independently for every session:
   $$\text{Bytes Read per Layer} \approx \underbrace{W_{\text{active}}(B)}_{\text{Shared Across Compatible Tokens}} + \underbrace{\sum_{r=1}^{B} S_r \times \text{KV}_{\text{bytes}}}_{\text{Private Per-Session Reads}}$$
   As sequence lengths $S_r$ grow, private KV reads dominate memory-bus utilization and delay weight-sharing stages.

3. **Machine Affinity ("Pegging"):**
   When an accelerator holds both layer weights and a session's KV cache, that session's decode loop is pinned to that physical machine across generation steps. Rebalancing load onto another replica requires either recomputing prefill or migrating substantial KV state over the network.

4. **Mixed-Length Contention & Multi-Layer Scheduling Collisions:**
   If a short request ($S = 512$) and a long request ($S = 64{,}000$) are coupled in lockstep, the short request's attention scan reads $125\times$ less KV data than the long request's scan, creating head-of-line waiting before FFN execution. Conversely, if requests advance asynchronously on a machine hosting multiple layers (e.g., Layers $0$–$7$), arrivals targeting Layer 6 and Layer 2 require distinct weight matrices and cannot coalesce into a single dense GEMM.

*(Notation Note — Global In-Flight Concurrency $M$ vs. Stage Batch Size $B$):* Throughout this paper, $M$ denotes the **global number of concurrent requests in flight** across a $P$-stage pipeline fabric (keeping $M \ge P$ forward micro-batches active simultaneously so stages remain occupied rather than stalling on pipeline bubbles), whereas $B$ denotes the **local batch size coalesced for a single execution step at an individual stage**. Because requests complete variable-length attention scans asynchronously, $M$ global users do not automatically produce a stage batch of size $B = M$; instead, $B$ depends on how many compatible requests arrive within a stage's coalescing window $\Delta t_{\text{wait}}$.

---

### 1.3 Positioning: Shared Foundations and Architectural Differences

Table 1.1 situates FrontierSplit relative to established attention kernels, disaggregated serving systems, pipeline runtimes, and sparse MoE engines:

#### Table 1.1: Comparison with Existing Inference Paradigms

| System / Paradigm | Shared Foundation | Architectural Difference in FrontierSplit |
| :--- | :--- | :--- |
| **FlashAttention-2 / FlashDecoding** *(Dao, 2023; Dao et al., 2023)* | Uses **Online Softmax** $(A_p, m_p, \ell_p)$ to tile sequence chunks in parallel and merge partial results via exact log-sum-exp rescaling. | FlashDecoding tiles within a **single accelerator's SMs**. FrontierSplit applies Online Softmax across a configurable SM interruption interval and, when needed, across **separate network hosts** over binary TCP. |
| **Phase-Disaggregated & Distributed KV Systems** *(DistServe, Zhong et al., 2024; Splitwise, Patel et al., 2024; Mooncake, Qin et al., 2024; PagedAttention/vLLM, Kwon et al., 2023)* | Separates prefill compute, decode compute, or distributed KV pools to avoid co-location bottlenecks and manage paged KV memory. | DistServe and Splitwise transfer generated KV caches from prefill instances to decode instances (**Data-to-Compute**), while Mooncake fetches KV blocks over RDMA/NICs. FrontierSplit keeps decode KV pinned on `Attn+KV` nodes (**Compute-to-Data**) and forwards `8–32 KiB` activations per layer step. |
| **Pipeline-Parallel Serving** *(Orca, Yu et al., 2022; Petals, Borzunov et al., 2023)* | Partitions vertical transformer layers across stages and exchanges `[B, d_model]` activations between stages. | Conventional pipeline stages co-locate each layer's KV cache with its FFN weights. FrontierSplit partitions *within* the layer into stateful `Attn+KV` nodes and **KV-stateless FFN/MoE workers**, separating per-session KV chunking from cross-request FFN batching. |
| **Attention-FFN Disaggregation (AFD)** *(vLLM `afd-plugin` & StepMesh, vLLM Project / StepFun, 2025–2026; SGLang AFD)* | Splits transformer blocks into separate `Attention+KV` and stateless `FFN/Expert` worker pools to decouple memory-bound KV scans from batched FFN/MoE compute. | Hyperscale AFD runtimes ping-pong dual micro-batches across all layers over **GPUDirect RDMA** (`StepMesh` / InfiniBand / RoCE) on frontier clusters. FrontierSplit targets **commodity/heterogeneous nodes over zero-pickle binary TCP**, combining vertical layer slicing for small-VRAM GPUs with **Online Softmax sequence chunking** and tunable interruption-interval striping. |
| **Sparse & MoE Execution Engines** *(DeepInfer, DeepSeek-AI, 2026; DeepSpeed-MoE, Rajbhandari et al., 2022; PowerInfer, Song et al., 2024)* | Optimizes intra-layer sparse/expert execution via token bucketing (`DeepGEMM`), expert dispatch (`DeepEP`/`EPLB`), or hot/cold neuron placement. | Orthogonal and complementary: FrontierSplit addresses **how unquantized models are carved across older/smaller VRAM budgets and how KV state is decoupled**, rather than modifying internal expert evaluation kernels. |

#### Scope & Operating Boundaries (When Not to Disaggregate)
Intra-layer disaggregation over commodity networking is **not** advantageous in every regime. When serving low concurrency ($M \le 4$) at moderate context lengths ($S \le 4{,}096$) on a high-bandwidth accelerator or tightly coupled multi-chip host where resident weights and KV caches fit comfortably in local HBM, adding TCP hops (`~60–80 us` per hop) increases latency without improving throughput. Our goal is to characterize the workload, capacity, and network conditions under which separating `Attn+KV` from `FFN` improves serving efficiency enough to justify additional communication and queueing—and when consolidated execution remains preferable.

---

## 2. Model, Layer, and KV Cache Anatomy (Unquantized 16-Bit Baseline)

All model capacity and tensor calculations in this paper target **unquantized 16-bit precision (`2 Bytes` per parameter: `BF16` on Ampere/Ada/Hopper/Blackwell/TPU, or `FP16` on Turing `NVIDIA T4`)**. To avoid conflating binary byte counts with decimal vendor link rates or marketing SKU labels:
* All calculated tensor sizes, layer weights, and KV footprints are stated in **exact binary units** (`1 KiB = 1,024 B`, `1 MiB = 1,048,576 B`, `1 GiB = 1,073,741,824 B`).
* Hardware memory-bus and network link rates retain standard vendor **decimal rates** (`GB/s = 10^9 B/s`, `Gbps = 10^9 bit/s`).
* Accelerator memory capacities distinguish **physical HBM/GDDR stack capacity** from **software-usable capacity** after ECC, row/column sparing, and driver reservations (detailed per SKU in Table 3.2).

### 2.1 Vertical Layer Counts, Resident Weights, and Dense vs. MoE Concurrency

A decoder-only LLM comprises $L$ vertical transformer blocks (in addition to the input token embedding table and final `lm_head` projection). Inside each layer $i \in \{0, \dots, L-1\}$:

1. **Attention Projections:**
   * $W_q \in \mathbb{R}^{d_{\text{model}} \times (H_q \cdot d_{\text{head}})}$, $W_k \in \mathbb{R}^{d_{\text{model}} \times (H_{\text{kv}} \cdot d_{\text{head}})}$, $W_v \in \mathbb{R}^{d_{\text{model}} \times (H_{\text{kv}} \cdot d_{\text{head}})}$, and $W_o \in \mathbb{R}^{(H_q \cdot d_{\text{head}}) \times d_{\text{model}}}$.
2. **Post-Attention FFN / MoE Block (SwiGLU):**
   * In **Dense** models, a single FFN block maps $d_{\text{model}} \to d_{\text{ffn}} \to d_{\text{model}}$ via three matrices ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$):
     $$\text{Bytes}_{\text{FFN}} = 3 \times d_{\text{model}} \times d_{\text{ffn}} \times 2\text{ Bytes}$$
   * In **Mixture-of-Experts (MoE)** models, the layer contains $E_{\text{total}}$ expert FFNs, activating $k_{\text{exp}} \ll E_{\text{total}}$ experts per token.

#### Resident Capacity vs. Active Execution: Dense as the Deterministic Cap
Because FrontierSplit focuses on **(a) carving unquantized models to fit older or smaller VRAM budgets** and **(b) decoupling KV state**, two distinct metrics govern FFN vs. MoE stages:
* **Physical VRAM Carving Uses Full Resident Weights:** Unless dynamically streaming cold experts from host DRAM, the accelerator(s) hosting an MoE layer must hold **all $E_{\text{total}}$ experts** in usable VRAM (`Table 2.1B`), regardless of how few experts fire on a single token.
* **Dense FFN Bounds Shared-Weight Concurrency ($B$), While MoE Adds Routing Stochasticity:** In a **Dense** FFN layer, every queued token multiplies against the exact same three matrices; Dense FFN thus serves as the **deterministic upper bound (cap)** on shared-weight batching efficiency. In an **MoE** layer, although $B = 1$ reads only $k_{\text{exp}} / E_{\text{total}}$ of the resident expert bytes, concurrent requests route **stochastically** across different experts. Because an individual SM or TensorCore executes one expert's matrix tile at a time, scattered routing means fewer tokens share any single loaded expert (prompting specialized MoE runtimes such as **DeepInfer** [DeepSeek-AI, 2026] to bucket incoming tokens per expert via grouped GEMMs before execution) and fewer simultaneous layers or expert tiles can co-execute across SMs/TensorCores before saturating memory bandwidth. Consequently, MoE layers realize lower effective per-expert batch reuse ($B_{\text{eff}} \le B$) than Dense layers of equivalent active size.

#### Table 2.1A: Open-Weight Dense Models (Unquantized 16-Bit Resident Layer Sizes)

| Model | Vertical Layers ($L$) | Hidden Dim ($d_{\text{model}}$) | Heads ($H_q / H_{\text{kv}} \times d_{\text{head}}$) | FFN Vertical Shape ($d_{\text{model}} \to d_{\text{ffn}}$) | Params / Layer | **Resident Layer Weight (16-bit)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Meta Llama 3.1 (8B)** | 32 | 4,096 | $32 / 8 \times 128$ | $4{,}096 \to 14{,}336 \to 4{,}096$ | 218.1M | **`416.0 MiB`** *(`0.41 GiB`)* |
| **Mistral 7B** | 32 | 4,096 | $32 / 8 \times 128$ | $4{,}096 \to 14{,}336 \to 4{,}096$ | 218.1M | **`416.0 MiB`** *(`0.41 GiB`)* |
| **Google Gemma 2 (9B)** | 42 | 3,584 | $16 / 8 \times 256$ | $3{,}584 \to 14{,}336 \to 3{,}584$ | 198.2M | **`378.0 MiB`** *(`0.37 GiB`)* |
| **Alibaba Qwen 2.5 (7B)** | 28 | 3,584 | $28 / 4 \times 128$ | $3{,}584 \to 18{,}944 \to 3{,}584$ | 233.0M | **`444.5 MiB`** *(`0.43 GiB`)* |
| **Mistral NeMo (12B)** | 40 | 5,120 | $32 / 8 \times 128$ | $5{,}120 \to 14{,}336 \to 5{,}120$ | 272.6M | **`520.0 MiB`** *(`0.51 GiB`)* |
| **Qwen 2.5 (14B / 32B)** | 48 / 64 | 5,120 | $40 / 8 \times 128$ | $5{,}120 \to 27{,}648 \to 5{,}120$ | 487.6M | **`930.0 MiB`** *(`0.91 GiB`)* |
| **Google Gemma 2 (27B)** | 46 | 4,608 | $32 / 16 \times 128$ | $4{,}608 \to 36{,}864 \to 4{,}608$ | 566.2M | **`1,080.0 MiB`** *(`1.05 GiB`)* |
| **Meta Llama 3.1 / 3.3 (70B)** | 80 | 8,192 | $64 / 8 \times 128$ | $8{,}192 \to 28{,}672 \to 8{,}192$ | 855.6M | **`1,632.0 MiB`** *(`1.59 GiB`)* |
| **Alibaba Qwen 2.5 (72B)** | 80 | 8,192 | $64 / 8 \times 128$ | $8{,}192 \to 29{,}568 \to 8{,}192$ | 877.7M | **`1,674.0 MiB`** *(`1.63 GiB`)* |
| **Meta Llama 3.1 (405B)** | 126 | 16,384 | $128 / 8 \times 128$ | $16{,}384 \to 53{,}248 \to 16{,}384$ | 3.19B | **`6,080.0 MiB`** *(`5.94 GiB`)* |

#### Table 2.1B: Mixture-of-Experts (MoE) Models (Resident Layer Size = All Experts Combined)

| Model | Vertical Layers ($L$) | $d_{\text{model}}$ | Expert Pool & Single Expert Shape ($d_{\text{model}} \to d_{\text{ffn}}$) | Single Expert Size | Params / Layer (Resident) | **Resident Layer Weight (16-bit)** |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **Mistral Mixtral 8x7B** *(top-2)* | 32 | 4,096 | **8 experts** ($4{,}096 \to 14{,}336 \to 4{,}096$) | `336.0 MiB` | 1.45B | **`2.70 GiB`** *(`2,768 MiB`)* |
| **Mistral Mixtral 8x22B** *(top-2)* | 56 | 6,144 | **8 experts** ($6{,}144 \to 16{,}384 \to 6{,}144$) | `576.0 MiB` | 2.50B | **`4.66 GiB`** *(`4,776 MiB`)* |
| **Databricks DBRX (132B)** *(top-4)* | 40 | 6,144 | **16 experts** ($6{,}144 \to 10{,}752 \to 6{,}144$) | `378.0 MiB` | 3.26B | **`6.07 GiB`** *(`6,216 MiB`)* |
| **xAI Grok-1 (314B)** *(top-2)* | 64 | 6,144 | **8 wide experts** ($6{,}144 \to 32{,}768 \to 6{,}144$) | `1.12 GiB` | 4.92B | **`9.16 GiB`** *(`9,384 MiB`)* |
| **DeepSeek-V3 / R1 (671B)** *(top-8 + 1)* | 61 *(3 dense + 58 MoE)* | 7,168 | **256 routed + 1 shared** ($7{,}168 \to 2{,}048 \to 7{,}168$) | `84.0 MiB` | 11.51B | **`21.43 GiB`** *(Dense: `1.09 GiB`)* |
| **OpenAI GPT-4 Turbo** *(Literature Est.)* | ~120 | ~12,288 | **16 experts** ($\approx 12{,}288 \to 32{,}768 \to 12{,}288$) | `~2.25 GiB` | ~15.0B | **`~28.0 GiB`** |
| **Alibaba Qwen-Max** *(Literature Est.)* | ~80–96 | ~8,192+ | **64–128 fine-grained + shared experts** | `~190–285 MiB` | ~25B–30B | **`~46.5 – 56.0 GiB`** |

---

### 2.2 The Intra-Layer Weight Asymmetry: Attention vs. Post-Attention FFN

Separating the **first sub-block of a transformer layer** (the Attention projections $W_q, W_k, W_v, W_o$) from the **second sub-block** (the Post-Attention FFN or resident MoE expert pool) highlights a consistent capacity asymmetry across model families:

#### Table 2.2: Intra-Layer Resident Weight Breakdown (16-Bit)

| Model | Attention $W_{q,k,v}$ Only | Full Attention ($W_{q,k,v,o}$) | Post-Attention FFN / Resident Experts | **Total Resident Layer** | **% of Resident Layer in FFN/MoE** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen 2.5 7B** | `31.5 MiB` | **`56.0 MiB`** | **`388.5 MiB`** | `444.5 MiB` | **87.4%** |
| **Llama 3.1 8B / Mistral 7B** | `48.0 MiB` | **`80.0 MiB`** | **`336.0 MiB`** | `416.0 MiB` | **80.8%** |
| **Gemma 2 9B** | `56.0 MiB` | **`84.0 MiB`** | **`294.0 MiB`** | `378.0 MiB` | **77.8%** |
| **Llama 3.1 70B** | `160.0 MiB` | **`288.0 MiB`** | **`1,344.0 MiB` (`1.31 GiB`)** | `1,632.0 MiB` (`1.59 GiB`) | **82.4%** |
| **Llama 3.1 405B** | `576.0 MiB` | **`1,088.0 MiB` (`1.06 GiB`)** | **`4,992.0 MiB` (`4.88 GiB`)** | `6,080.0 MiB` (`5.94 GiB`) | **82.1%** |
| **Mixtral 8x7B (MoE)** | `48.0 MiB` | **`80.0 MiB`** | **`2,688.0 MiB` (`2.62 GiB`)** | `2,768.0 MiB` (`2.70 GiB`) | **97.1%** |
| **DeepSeek-V3 / R1 (MoE)** | *(MLA projections)* | **`~356.7 MiB`** | **`21,588.0 MiB` (`21.08 GiB`)** | `~21,945 MiB` (`21.43 GiB`) | **98.4%** |

**Implication for Partitioning:** Across `7B`–`70B` dense models and `Mixtral`/`DeepSeek` MoEs, full attention projections occupy **`56 MiB` to `357 MiB` per layer** (reaching `1,088 MiB` = `1.06 GiB` at the `405B` scale), whereas **$77.8\%$ to $98.4\%$ of resident layer capacity** sits in the post-attention FFN or expert weights, which do not read or store KV state. *(Even for per-step active bytes at $B = 1$, Mixtral's 2 active experts read `672 MiB` vs. `80 MiB` in attention = `89.4%` FFN, and DeepSeek-V3's 9 active experts read `756 MiB` vs. `~357 MiB` in MLA = `67.9%` FFN, rising toward resident share as $B$ increases across scattered experts).*

---

### 2.3 Per-Layer and Full-Model KV Cache Sizing Across Sequence Lengths ($S$)

Llama 405B ($d_{\text{model}} = 16{,}384$) has the same per-layer KV cache footprint as Llama 8B ($d_{\text{model}} = 4{,}096$) because $d_{\text{model}}$ governs the hidden state width between layers, whereas Grouped-Query Attention (GQA) projects Keys and Values down to $H_{\text{kv}} = 8$ heads of width $d_{\text{head}} = 128$ prior to caching:

$$\text{KV Bytes per Token per Layer} = 2\ (\text{K \& V}) \times H_{\text{kv}} \times d_{\text{head}} \times 2\text{ Bytes}$$

* **Standard GQA ($H_{\text{kv}} = 8, d_{\text{head}} = 128$ — Llama 3.1 8B/70B/405B, Qwen 2.5 14B/32B/72B, Mixtral):**
  $$2 \times 8 \times 128 \times 2\text{ Bytes} = \mathbf{4{,}096\text{ Bytes (`4.00 KiB`) per token per layer}}$$
* **Compressed Multi-Head Latent Attention (MLA — DeepSeek-V3/R1):**
  Compresses KV heads into a shared latent vector of $512$ elements plus $64$ RoPE positional elements:
  $$(512 + 64) \times 2\text{ Bytes} = \mathbf{1{,}152\text{ Bytes (`1.125 KiB`) per token per layer}}$$
* **Multi-Query Attention (MQA, $H_{\text{kv}} = 1$ — Falcon, StarCoder):**
  $$2 \times 1 \times 128 \times 2\text{ Bytes} = \mathbf{512\text{ Bytes (`0.50 KiB`) per token per layer}}$$
* *(Hybrid architectures such as Gemma 2 alternate full-context layers with `4K` sliding-window layers; Mamba/SSM hybrids allocate fixed-size recurrent state rather than growing KV caches on SSM layers).*

#### Table 2.3: Per-Layer vs. Full-Model KV Cache Footprint (16-Bit, Single Session)

| Sequence Length ($S$) | **Single Layer KV** <br>*(GQA: `4.00 KiB/tok`)* | **Single Layer KV** <br>*(MLA: `1.125 KiB/tok`)* | Full Model: **Llama 8B** <br>*(32 Layers)* | Full Model: **Llama 70B** <br>*(80 Layers)* | Full Model: **Llama 405B** <br>*(126 Layers)* | Full Model: **DeepSeek-V3** <br>*(61 Layers, MLA)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **512 tokens** | **`2.00 MiB`** | **`0.56 MiB`** | `64.0 MiB` | `160.0 MiB` | `252.0 MiB` | `34.3 MiB` |
| **4,096 (4K)** | **`16.00 MiB`** | **`4.50 MiB`** | `512.0 MiB` (`0.50 GiB`) | `1.25 GiB` | `1.97 GiB` | `274.5 MiB` (`0.27 GiB`) |
| **8,192 (8K)** | **`32.00 MiB`** | **`9.00 MiB`** | `1.00 GiB` | `2.50 GiB` | `3.94 GiB` | `549.0 MiB` (`0.54 GiB`) |
| **32,768 (32K)** | **`128.00 MiB`** (`0.125 GiB`) | **`36.00 MiB`** (`0.035 GiB`) | `4.00 GiB` | `10.00 GiB` | `15.75 GiB` | `2.14 GiB` |
| **128K (131,072)** | **`512.00 MiB`** (`0.50 GiB`) | **`144.00 MiB`** (`0.14 GiB`) | `16.00 GiB` | `40.00 GiB` | `63.00 GiB` | `8.58 GiB` |
| **256K (262,144)** | **`1.00 GiB`** | **`288.00 MiB`** (`0.28 GiB`) | `32.00 GiB` | `80.00 GiB` | `126.00 GiB` | `17.16 GiB` |
| **512K (524,288)** | **`2.00 GiB`** | **`576.00 MiB`** (`0.56 GiB`) | `64.00 GiB` | `160.00 GiB` | `252.00 GiB` | `34.31 GiB` |
| **1,024K (1M)** | **`4.00 GiB`** | **`1.125 GiB`** | `128.00 GiB` | `320.00 GiB` | `504.00 GiB` | `68.62 GiB` |

**Single-Session vs. Multi-Session Layer Capacity:** In standard GQA, every $256\text{K}$ tokens ($262{,}144$ tokens) equals **`1.00 GiB` per layer** (`288.0 MiB` in MLA). For an **individual user session**, even a `1M`-token context occupies `4.00 GiB` in a single layer—meaning one session's single-layer KV state fits on a single GPU without cross-machine sharding unless an operator chooses to parallelize memory bandwidth via Tier 3 Online Softmax fan-out. Across **$M_{\text{active}}$ concurrent long-context sessions**, however, aggregate per-layer KV state scales as $M_{\text{active}} \times \text{KV}_{\text{session}}$ (for example, $64$ sessions at $1\text{M}$ tokens require $256\text{ GiB}$ for Layer $i$ alone), which is distributed horizontally by assigning distinct user sessions across a pool of `Attn+KV` hosts for Layer $i$.

---

### 2.4 Data Transferred Between Stages During Decode

During autoregressive decode, each active session advances by $1$ token per step. The hidden-state activation passed between sub-layers or layers has shape `[1, d_model]` (matching the element count of `[H_q, 128, 1]` when $d_{\text{head}} = 128$) and is independent of prompt length $S$:

#### Table 2.4: Per-Token Activation Payloads (16-Bit)

| Model Tier | Hidden Dim ($d_{\text{model}}$) | Equivalent Shape `[H_q, 128, 1]` | **Single-Token Payload (16-bit)** | Coalesced Stage Payload ($B$ tokens) |
| :--- | :---: | :---: | :---: | :---: |
| **8B / Mixtral Tier** *(Llama 8B, Mixtral 8x7B)* | 4,096 | `[32, 128, 1]` | **`8,192 Bytes` (`8.00 KiB`)** | $8.00\text{ KiB} \times B$ |
| **DeepSeek-V3 / R1** | 7,168 | `[56, 128, 1]` | **`14,336 Bytes` (`14.00 KiB`)** | $14.00\text{ KiB} \times B$ |
| **70B Tier** *(Llama 70B, Qwen 72B)* | 8,192 | `[64, 128, 1]` | **`16,384 Bytes` (`16.00 KiB`)** | $16.00\text{ KiB} \times B$ |
| **405B Tier** *(Llama 405B)* | 16,384 | `[128, 128, 1]` | **`32,768 Bytes` (`32.00 KiB`)** | $32.00\text{ KiB} \times B$ |

*(Note: When merging remote KV partitions prior to $W_o$ projection under Tier 3 multi-node fan-out, transmitting unnormalized FP32 Online Softmax accumulators $(A_p, m_p, \ell_p)$ for a 32-head model requires $32 \times 128 \times 4 + 2 \times 32 \times 4 = 16{,}640\text{ Bytes}$ (`16.25 KiB`) per shard, whereas forwarding the post-$W_o$ residual hidden state $h_i$ in Option C requires `8.00 KiB` in 16-bit precision).*

---

## 3. Datacenter Transport Tiers & Accelerator Execution Mechanics

### 3.1 Datacenter Transport Latency, Serialization Bandwidth, and Layer Fan-Out

One-way message transfer time across any interconnect includes both fixed protocol/switch overhead and wire serialization time:

$$T_{\text{transfer}}(P) = T_{\text{base\_latency}} + \frac{P}{\text{Bandwidth}_{\text{link}}}$$

* **Single-Token Messages Entering/Leaving `Attn+KV` Nodes:** Because each KV cache belongs to a distinct session, requests enter `Attn+KV` nodes as single-token messages ($P = 8\text{–}32\text{ KiB}$). Even after $B$ sessions coalesce at a stateless `FFN` node to share weight loads, at the completion of that `FFN` layer the batch **splits (fans out)** back into $B$ individual 1-token messages routed to each session's respective next-layer `Attn+KV` host. For an isolated `32 KiB` (`262,144-bit`) packet on a `100 Gbps` NIC, wire serialization is `2.62 us`, so fixed socket and switch latency (`~60–80 us` on intra-AZ TCP) dominates single-packet transit.
* **Aggregate Fan-Out & Coalesced Inter-Stage Transfers ($B \gg 1$):** Link bandwidth becomes significant when (1) an `FFN` node fans out $B$ single-token packets concurrently over the same physical NIC, or (2) two consecutive multi-layer `FFN` stages exchange a coalesced batch tensor of size $P = B \times d_{\text{model}} \times 2\text{ Bytes}$ directly. Transmitting `64` tokens at $d_{\text{model}} = 16{,}384$ (`2.00 MiB` = `16.78 Mbit`) adds `167.8 us` of serialization time on `100 Gbps`, or `1.68 ms` on `10 Gbps`. We therefore include $\frac{P}{\text{Bandwidth}_{\text{link}}}$ in all analytical and simulation models.

#### Table 3.1: Datacenter Interconnect Tiers (Isolated `8–32 KiB` One-Way Transit)

| Transport Tier | Technology / Protocol | Typical Availability | Isolated `8–32 KiB` Transit | Relative Latency |
| :--- | :--- | :--- | :---: | :---: |
| **Tier 1: On-Board / Pod Fabric** | **NVIDIA NVLink** / **Google TPU ICI** | Multi-GPU chassis / NVL72 rack or TPU Pod slice (direct memory/optical fabric) | **`~1.0 – 2.0 us`** <br>*(0.001–0.002 ms)* | **1x (Baseline)** |
| **Tier 2: Kernel-Bypass RDMA** | **AWS EFA (SRD)** / **GCP GPUDirect-TCPX** / **InfiniBand** | High-end GPU instances (`p4d`, `p5`, `a3`) within placement groups | **`~12 – 25 us`** <br>*(0.012–0.025 ms)* | **~15x** |
| **Tier 3A: Raw Binary TCP (`asyncio`)** | **Persistent POSIX Socket + `TCP_NODELAY`** | Commodity & frontier Cloud VMs in the same Availability Zone | **`~60 – 80 us`** <br>*(0.060–0.080 ms)* | **~50x** |
| **Tier 3B: Framed HTTP/RPC (gRPC)** | **HTTP/2 + Protobuf Serialization** | Standard Cloud VMs | **`~150 – 350 us`** <br>*(0.150–0.350 ms)* | **~200x** |

#### Choosing Persistent Binary TCP vs. NCCL Point-to-Point
Although NVIDIA NCCL provides point-to-point `ncclSend`/`ncclRecv` primitives in addition to collective operations (NVIDIA, 2024a), NCCL communicators bind a statically initialized set of homogeneous CUDA ranks and share communicator-wide error state if a participating rank fails or scales dynamically. Because FrontierSplit targets **heterogeneous cloud pools**—including commodity PCIe GPU VMs without GPUDirect RDMA, CPU-backed or lower-tier `Attn+KV` hosts paired with GPU/TPU `FFN` workers, and independent per-stage autoscaling—we use persistent framed binary TCP (`TCP_NODELAY` with zero-copy buffer views) for inter-node routing (`~60–80 us` intra-AZ base latency), while retaining intra-host NVLink or TPU ICI within multi-chip stages where applicable.

---

### 3.2 Accelerator Silicon Hierarchy, SKU Capacities, and $[N \times 128]$ Normalization

Sections 3.2 and 3.3 express standard accelerator hardware specifications in a unified notation for cross-platform comparison. Because boost clocks across datacenter accelerators span roughly a $2\times$ range (`1.05 GHz` on TPU v4 to `2.04 GHz` on NVIDIA L4), peak throughput differences stem primarily from **spatial parallelism**—the count and organization of 16-bit Multiply-Accumulate (MAC) units:

1. **Level 1 — Compute Clusters on the Die:** Streaming Multiprocessors (**SMs**) on NVIDIA GPUs, or **TensorCores** on Google TPUs.
2. **Level 2 — Matrix Engines Within Each Cluster:** Warp-driven Tensor Cores on GPUs, or systolic Matrix Multiply Units (**MXUs**) on TPUs (for example, both Google Cloud TPU v4 and TPU v5p house **2 TensorCores per chip** and **4 MXUs per TensorCore** = **8 MXUs per chip**; Google Cloud, 2024a, 2024b).
3. **Level 3 — Native Multiplier Array:** The 2D array of 16-bit MAC units inside each engine ($128 \times 128 = 16{,}384$ MACs per TPU MXU).

Because most modern transformer families use an attention head dimension of **$d_{\text{head}} = 128$** (with occasional multiples such as $d_{\text{head}} = 256$ in Gemma 2 9B, spanning two 128-wide slices) and hidden dimensions $d_{\text{model}}$ divisible by 128, we write any chip's active 16-bit MAC count in units of **128-element inner products**:

$$\text{Normalized MAC Capacity} = N \times 128, \qquad \text{where } N = \frac{\text{Total Active 16-bit MACs}}{128}$$

Table 3.2 reports exact primary-source SKUs alongside **both physical stack memory and software-usable memory** (accounting for ECC, row/column sparing, and firmware reservations—most notably on `NVIDIA HGX B200`, where `8 x 24 GiB = 192 GiB` physical HBM3e stacks expose **up to `180 GB`** of usable memory to CUDA applications; NVIDIA, 2025):

#### Table 3.2: Primary-Source Accelerator Organization, Usable Memory, & $[N \times 128]$ Capacity

| Accelerator SKU | **Physical / Usable Memory** | **Level 1: Active Clusters** | **Level 2: Engines / Cluster** | **Level 3: 16-bit MACs / Engine** | **Total Active 16-bit MACs** | **Normalized ($N \times 128$)** | Peak Dense 16-bit TFLOPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA T4 PCIe** *(FP16)* | `16 GiB` / `~15.0 GiB` | **40 SMs** | 8 Tensor Cores/SM | 64 MACs | **20,480** | **$160 \times 128$** | **65 TFLOPS** |
| **NVIDIA L4 PCIe** | `24 GiB` / `~22.5 GiB` | **58 SMs** | 4 Tensor Cores/SM | 128 MACs | **29,696** | **$232 \times 128$** | **121 TFLOPS** |
| **Google TPU v4** *(1 MXU)* | *(Sub-unit)* | *(Sub-unit)* | 1 MXU | 16,384 MACs ($128^2$) | **16,384** | **$128 \times 128$** | **~34.4 TFLOPS** |
| **Google TPU v4** *(1 Chip)* | `32 GiB` / `32.0 GiB` | **2 TensorCores** | **4 MXUs/TensorCore** | 16,384 MACs ($128^2$) | **131,072** | **$1{,}024 \times 128$** | **275 TFLOPS** |
| **Google TPU `v4-8`** *(4 Chips)* | `128 GiB` / `128.0 GiB` | **8 TensorCores** | **4 MXUs/TensorCore** | 16,384 MACs ($128^2$) | **524,288** | **$4{,}096 \times 128$** | **1,100 TFLOPS** |
| **NVIDIA A100 SXM4** *(80GB)* | `80 GiB` / `~79.2 GiB` | **108 SMs** | 4 Tensor Cores/SM | 256 MACs | **110,592** | **$864 \times 128$** | **312 TFLOPS** |
| **NVIDIA H100 SXM5** *(80GB)* | `80 GiB` / `~79.2 GiB` | **132 SMs** | 4 Tensor Cores/SM | 512 MACs | **270,336** | **$2{,}112 \times 128$** | **989 TFLOPS** |
| **Google TPU `v5p-8`** *(4 Chips)* | `380 GiB` / `380.0 GiB` | **8 TensorCores** | **4 MXUs/TensorCore** | 16,384 MACs ($128^2$) | **524,288** | **$4{,}096 \times 128$** | **1,836 TFLOPS** |
| **NVIDIA HGX B200** *(1000W)* | `192 GiB` / **`≤180 GB`** | **148 SMs** *(of 160)* | 4 Tensor Cores/SM | 1,024 MACs | **606,208** | **$4{,}736 \times 128$** | **2,250 TFLOPS** |

*(Note on Google Cloud TPU Naming: The suffix in `v4-8`, `v4-16`, or `v5p-8` counts total **TensorCores** across the slice; at 2 TensorCores and 8 MXUs per chip, `v4-8` comprises **4 chips = 32 MXUs = `128.0 GiB` HBM**, and `v4-16` comprises **8 chips = 64 MXUs = `256.0 GiB` HBM**).*

#### Two Standard Tiling Orientations for `[H, 128, S]`
For an attention tensor shaped `[Heads, 128, Tokens]` (e.g., $H = 32, d_{\text{head}} = 128$) mapped onto a $16{,}384$-MAC group:
* **Orientation 1 — Cross-Head Packing (`[32, 128, 4]`):** Stages $Q$ across 32 heads (`[32, 128, 1]`) and evaluates **4 historical token positions across all 32 heads** ($32 \times 128 \times 4 = 16{,}384$ MACs), naturally exploiting GQA head sharing where multiple Query heads read one KV head.
* **Orientation 2 — Sequence-Contiguous Tiling (`[1, 128, 128]`):** Stages $Q$ for a single head (`[1, 128, 1]`) and evaluates **128 contiguous historical tokens** ($1 \times 128 \times 128 = 16{,}384$ MACs).

---

### 3.3 Weight-Only Roofline Crossover ($B_{\text{crit}}$), Tile-Refill Intuition, and Memory Occupancy

#### 1. Analytical Weight-Only Crossover ($B_{\text{crit}}$)
Consider a dense 16-bit (`2-byte`) matrix multiplication $[B, d_{\text{in}}] \times [d_{\text{in}}, d_{\text{out}}]$. Assuming the weight matrix is read once from off-chip memory and reused across all $B$ inputs—and neglecting input/output activation traffic ($2B(d_{\text{in}} + d_{\text{out}}) \ll 2 d_{\text{in}} d_{\text{out}}$ when $B \ll d_{\text{in}}, d_{\text{out}}$)—the idealized arithmetic intensity is:

$$I(B) \approx \frac{2 B \, d_{\text{in}} d_{\text{out}}\ \text{FLOPs}}{2 \, d_{\text{in}} d_{\text{out}}\ \text{Bytes}} = B\ \text{FLOP/Byte}$$

Equating $I(B)$ to the accelerator's peak dense 16-bit compute-to-bandwidth ratio gives the **idealized weight-only batch crossover**:

$$B_{\text{crit}} \approx \frac{F_{\text{peak}}}{\text{BW}_{\text{peak}}}$$

#### Table 3.3: Primary-Source Roofline Ridge Points and Idealized Batch Crossover ($B_{\text{crit}}$)

| Accelerator SKU | Peak Dense 16-Bit Compute ($F_{\text{peak}}$) | Peak Memory Bandwidth ($\text{BW}_{\text{peak}}$) | **Ideal Weight-Only Crossover $B_{\text{crit}} = F_{\text{peak}} / \text{BW}_{\text{peak}}$** |
| :--- | :---: | :---: | :---: |
| **NVIDIA A100 SXM4** *(80GB)* | 312 TFLOPS | 2,039 GB/s | **~153.0 FLOP/Byte** *($B_{\text{crit}} \approx 153$)* |
| **Google TPU v5p** *(1 Chip, 95 GiB)* | 459 TFLOPS | 2,765 GB/s | **~166.0 FLOP/Byte** *($B_{\text{crit}} \approx 166$)* |
| **NVIDIA T4 PCIe** *(16GB, FP16)* | 65 TFLOPS | 320 GB/s *(300+ GB/s rated)* | **~203.1 FLOP/Byte** *($216.7$ at $300\text{ GB/s}$)* |
| **Google TPU v4** *(1 Chip, 32 GiB)* | 275 TFLOPS | 1,200 GB/s | **~229.2 FLOP/Byte** *($B_{\text{crit}} \approx 229$)* |
| **NVIDIA HGX B200** *(180GB usable)* | 2,250 TFLOPS | 8,000 GB/s | **~281.3 FLOP/Byte** *($B_{\text{crit}} \approx 281$)* |
| **NVIDIA H100 SXM5** *(80GB)* | 989 TFLOPS | 3,350 GB/s | **~295.2 FLOP/Byte** *($B_{\text{crit}} \approx 295$)* |
| **NVIDIA L4 PCIe** *(24GB)* | 121 TFLOPS | 300 GB/s | **~403.3 FLOP/Byte** *($B_{\text{crit}} \approx 403$)* |

#### 2. Microarchitectural Intuition ($\lfloor C / k \rfloor$) and Practical Qualifications
At the tile level, $B_{\text{crit}}$ can be visualized heuristically through double-buffered staging: if fetching the next weight tile from VRAM into local staging SRAM requires an idealized window of $C$ cycles while multiplying one input against the currently resident weight tile requires $k$ idealized MAC cycles, then $\lfloor C / k \rfloor = B_{\text{crit}}$ (by construction from $F_{\text{peak}} / \text{BW}_{\text{peak}}$, not as an independent hardware timing law) represents how many batched inputs could ideally execute before the incoming weight tile is ready to overwrite the staged buffer:

* **Below Crossover ($1 \le B \ll B_{\text{crit}}$):** Execution is **memory-bandwidth dominated**; increasing $B$ reuses resident weights and increases total step time much more slowly than $B \times$ sequential single-token calls.
* **Above Crossover ($B > B_{\text{crit}}$):** Larger batch sizes remain completely valid, but matrix compute runs **past** the minimum VRAM fetch window, entering the **compute-dominated** regime where step latency grows roughly linearly with $B$.
* **Important Qualifications:** In real kernels, $B_{\text{crit}}$ is a **soft transition rather than a sharp cycle boundary**. Actual crossover shifts downward or smooths out due to finite matrix shapes and tile quantization, achieved Tensor Core / MXU utilization below $F_{\text{peak}}$, input/output activation reads and writes, L2 cache hit behavior, register pressure, and kernel launch overhead. We therefore treat $B_{\text{crit}}$ strictly as an analytical reference scale to test empirically.

#### 3. Chip-Wide Occupancy and Memory-Level Parallelism
Achieving a chip's rated memory bandwidth requires maintaining high **memory-level parallelism**—enough concurrent memory transactions in flight across the die to hide DRAM access latency (Little's Law) and drive all memory controllers (and on dual-TensorCore TPUs, both TensorCores' HBM interfaces). Although GPU SMs route across memory partitions via the L2 crossbar rather than being hard-wired to isolated DRAM slices, a workload confined to one warp or a small handful of SMs typically cannot issue enough outstanding requests to saturate chip-wide HBM/GDDR bandwidth. Saturating memory bandwidth therefore favors keeping SMs/TensorCores occupied across the chip—either via intra-request chunk parallelism or concurrent multi-request scheduling.

---

## 4. Disaggregated Pipeline Topologies & Three-Tier Chunking

### 4.1 Multi-Layer Out-and-Back Contention

Consider a worker hosting multiple consecutive layers (e.g., Layers $0$–$7$) while offloading KV state to remote KV servers. Interleaving concurrent requests on that worker introduces two structural inefficiencies:

```
TIME (ms)   MODEL WORKER (Holds Layers 0–7)           REMOTE KV(0) NODE
─────────   ───────────────────────────────           ─────────────────
0.00 ms     Runs Request 1 (Layer 0 QKV-proj)
0.05 ms     Sends Q1 to KV(0) ──────────────────────► Scans KV(0) for Req 1
            [Worker switches to Request 2]            (~0.20 ms)
0.05 ms     Starts Request 2 (e.g., Layer 3 FFN)...
0.25 ms     [QUEUE CONTENTION]                        ◄── Req 1 returns from KV(0)
            Req 1 is ready for Layer 0 FFN,
            but Worker is executing Layer 3 for Req 2
```

1. **Repeated Out-and-Back Hop Latency:** Across 8 layers on the same worker, a token makes 8 round-trip RPCs (`16` one-way network transits $\approx 1.0\text{–}1.3\text{ ms}$ of base TCP latency per layer group).
2. **Cross-Layer Queue Fragmentation:** Returning attention outputs for Layer 0 contend with requests executing Layers $1$–$7$. Because different layers require distinct weight matrices, requests sitting at different layer indices cannot coalesce into a shared-weight GEMM.

---

### 4.2 Intra-Layer Dataflow: Comparing Three Partition Topologies

Inside a single transformer layer $i$, the forward computation (with lightweight RMSNorm vectors of size `d_model` = `8–32 KiB` kept local to the consumer projection) is:

$$\text{Input } x_i \xrightarrow{W_{q,k,v}} (Q_i, K_i, V_i) \xrightarrow{\text{Attention with KV}_i} \text{Attn}_i \xrightarrow{W_o\ +\ \text{Residual } x_i} h_i \xrightarrow{\text{FFN}_i\ +\ \text{Residual } h_i} x_{i+1}$$

Accounting explicitly for residual paths ($x_i$ and $h_i$), newly generated $(K_i, V_i)$ vectors, and output projection $W_{o,i}$ yields **three candidate distributed partitions** for Layer $i$:

```
OPTION A: Out-and-Back (Full-Layer Worker + Weightless KV Node)
┌───────────────────────────┐   1. Send Q_i, K_i, V_i   ┌───────────────────────────┐
│      Model Worker (i)     │ ────────────────────────► │        KV Node (i)        │
│ Holds: W_qkv, W_o, FFN_i  │ ◄──────────────────────── │ Holds: 0 Weights, KV_i    │
└─────────────┬─────────────┘   2. Return (A_p, m_p, l) └───────────────────────────┘
              │ 3. Finish W_o + x_i + FFN_i, send x_{i+1} (8–32 KiB)
              ▼
┌───────────────────────────┐
│     Model Worker (i+1)    │
└───────────────────────────┘

OPTION B: Half-Layer Staggered Pipeline (Forward-Only, Weightless KV Node)
┌───────────────────────────┐   1. Send Q_i, K_i, V_i   ┌───────────────────────────┐
│     Stage Worker (i)      │ ────────────────────────► │        KV Node (i)        │
│ Holds: FFN_{i-1} + W_qkv_i│ ────┐                     │ Holds: 0 Weights, KV_i    │
└───────────────────────────┘     │ Pass residual x_i   └─────────────┬─────────────┘
                                  ▼                                   │ 2. Forward (A_p, m_p, l)
                            ┌─────────────────────────────────────────▼─────────────┐
                            │                   Stage Worker (i+1)                  │
                            │       Holds: W_o_i + FFN_i  AND  W_qkv_{i+1}          │
                            └───────────────────────────────────────────────────────┘

OPTION C: Co-Located Attention+KV Node ──► KV-Stateless FFN Node (Forward Split)
┌───────────────────────────────────────────────────────┐
│            Attention + KV Node (i)                    │
│  • Resident Weights: W_q, W_k, W_v, W_o (56–1088 MiB) │
│  • Dynamic State:    Layer i KV Cache (1 GiB / 256K)  │
│  1. Receives single-token x_i (8–32 KiB)              │
│  2. Computes Q_i, K_i, V_i locally                    │
│  3. Appends K_i, V_i to local ragged-edge KV chunk    │
│  4. Runs Attention + Softmax + W_o + adds residual x_i│
└───────────────────────────┬───────────────────────────┘
                            │ 1 Hop: Sends h_i (8–32 KiB) forward
                            ▼
┌───────────────────────────────────────────────────────┐
│          KV-Stateless FFN / Expert Node (i)           │
│  • Resident Weights: FFN_i / MoE Experts (78%–98.4%)  │
│  • Dynamic State:    No KV Cache                      │
│  1. Coalesces ready h_i arrivals (window Δt_wait)     │
│  2. Runs FFN_i + adds residual h_i                    │
│  3. Fans out 1-token x_{i+1} (8–32 KiB) to Attn+KV(i+1)│
└───────────────────────────────────────────────────────┘
```

#### Table 4.1: Per-Layer Partition Comparison (Single-Shard KV Case)

| Metric | **Option A: Out-and-Back** | **Option B: Half-Layer Staggered** | **Option C: Co-Located `Attn+KV` $\to$ `FFN`** |
| :--- | :--- | :--- | :--- |
| **Critical-Path Network Hops / Layer** | **3 sequential hops** (`Model(i) -> KV(i) -> Model(i) -> Model(i+1)`) | **2 sequential hops** (`Stage(i) -> KV(i) -> Stage(i+1)`; residual $x_i$ sent in parallel) | **2 sequential hops** (`Attn_KV(i) -> FFN(i) -> Attn_KV(i+1)`) |
| **Messages Sent / Layer / Token** | **3 messages** | **3 messages** (`QKV`, `(A,m,l)`, plus parallel residual `x_i`) | **2 messages** (`h_i` and `x_{i+1}`) |
| **Wire Bytes / Layer (Llama 70B, GQA)** | `20 KiB` (`QKV`) + `32.5 KiB` (`FP32 A,m,l`) + `16 KiB` (`x_{i+1}`) = **`68.5 KiB`** | `20 KiB` + `32.5 KiB` + `16 KiB` (`x_i`) = **`68.5 KiB`** | `16 KiB` (`h_i`) + `16 KiB` (`x_{i+1}`) = **`32.0 KiB`** |
| **Dataflow Direction** | Bi-directional (returns to `Model(i)`) | Unidirectional forward | Unidirectional forward |
| **New Token $(K_i, V_i)$ Append** | Transmitted over network to `KV(i)` | Transmitted over network to `KV(i)` | **Projected & chunked locally in `Attn_KV(i)` memory** |
| **Weights on KV Node** | `0 MiB` | `0 MiB` | **`56–357 MiB`** (`1,088 MiB` on `405B`) |
| **FFN / MoE Worker Queueing** | Shares worker queue with pre-attention $W_{q,k,v}$ | Coupled with next layer's $W_{q,k,v}$ | **Dedicated queue for `FFN_i` only; coalesces over window $\Delta t_{\text{wait}}$** |

**Trade-Off Summary:** Option C places $W_{q,k,v,o}$ (`80 MiB` on Llama 8B/Mixtral, `288 MiB` on Llama 70B) alongside the KV cache. Doing so avoids the parallel residual transfer required in Option B, cuts per-layer wire volume by more than half (`32.0 KiB` vs. `68.5 KiB` on 70B), keeps new-token KV updates and chunk migration local to `Attn+KV(i)`, and isolates `FFN(i)` into a single-stage queue. Relative to a consolidated co-located layer, Option C introduces two sequential network hops per disaggregated layer boundary plus any batch-coalescing wait $\Delta t_{\text{wait}}$ incurred at `FFN(i)` when global concurrency $M$ is sparse. *(When usable VRAM permits, consecutive layers can also be grouped into coarser multi-layer blocks to reduce total hop count).*

---

### 4.3 Multi-Query Attention vs. Sequence Chunking, Ragged-Edge Alignment, and Striping

Each active session contributes one query vector per decode iteration. An `Attn+KV` node *can* batch queries from multiple ready sessions during the shared $W_{q,k,v,o}$ projections and even launch a multi-session paged attention kernel, **but private historical KV data must still be read separately from memory for each session**—so batching across sessions does not yield the shared-memory reuse of dense FFN batching. **Chunking**, by contrast, partitions an *individual* session's KV scan along the sequence dimension $S$. In FrontierSplit, we structure KV attention execution around sequence chunking across three levels (choosing per-chunk scheduling so mixed-length sessions do not block one another in lockstep):

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 3: MULTI-NODE KV CHUNKING (Cross-Host Online Softmax Fan-Out)                     │
│   • Invoked when a single session's S warrants multi-host memory bandwidth or capacity │
│   • Combines partial (A_p, m_p, ℓ_p) states via algebraically exact FP32 rescaling     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 2: SM / TENSORCORE CHUNKING (Tunable Interruption Interval & Striping Unit)       │
│   • Software workaround for lack of mid-kernel hardware preemption on GPUs/TPUs        │
│   • Self-Striping (low load): spreads one session's chunks across chip SMs/TensorCores │
│   • Concurrency-Striping (contended): multiplexes chunks of concurrent sessions        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 1: WARP / MXU TILE CHUNKING (Aligned History Blocks + Active Ragged-Edge Chunk)   │
│   • Committed KV history ends on exact chunk boundaries                                │
│   • Partial tail ("ragged edge") runs as one chunk alongside new token; migrates full  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 1. Tier 1: Aligned KV Chunks and Local "Ragged-Edge" Management
At the lowest level, historical KV tokens are stored in fixed-width chunks aligned to native hardware tile multiples (`16` to `128` tokens).
* **Ragged-Edge Tail Handling on `Attn+KV` Nodes:** Committed KV cache blocks end on clean chunk boundaries. Tokens generated since the last full boundary—the **ragged edge**—reside in an active tail buffer loaded into the attention block alongside the newly projected $(Q_t, K_t, V_t)$. Even when partially populated (with causal masking over unused slots), this tail executes structurally as one standard chunk. Once newly generated tokens fill the active tail to a full chunk boundary, the completed block is committed to the aligned KV store and a fresh tail chunk is allocated.
* **Prologue and Epilogue Overhead:** Each chunk execution incurs a setup cost to stage $Q$ into warp/MXU registers, initialize partial Online Softmax accumulators $(A_p, m_p, \ell_p)$, and write out partial states for reduction.

#### 2. Tier 2: The Interruption-Interval Knob & Two Modes of Striping
Once a thread block or MXU wave begins executing a chunk on an SM or TensorCore, **current GPU and TPU hardware does not provide fine-grained mid-kernel preemption** to pause that chunk for a newly arriving request. We work around this hardware constraint in software by exposing the **Tier 2 sequence-chunk quantum (the interruption interval)** as a **tunable configuration knob**:

* **Two Forms of Striping:**
  1. **Self-Striping (Intra-Request Parallelism):** When an `Attn+KV` node is otherwise uncontested, a single session's KV chunks stripe across available SMs/TensorCores to build sufficient memory-level parallelism across the chip, followed by a local Online Softmax reduction. *(If a session's context $S$ is short—e.g., $S = 512$—it occupies only enough SMs to amortize chunk prologue overhead, leaving remaining SMs available).*
  2. **Concurrency-Striping (Inter-Request Multiplexing):** When multiple sessions residing on the same `Attn+KV` node receive queries concurrently, dispatching a `128K`-token KV scan as a single uninterruptible kernel would occupy SMs until completion. Partitioning long scans into bounded **interruption-interval** quanta (e.g., `1,024` to `8,192` tokens per dispatch) creates cooperative scheduling boundaries between quanta where chunks from newly arrived short requests can interleave across SMs/TensorCores or concurrent hardware streams.
* **Parameterizing the Knob:** Larger interruption intervals reduce kernel-launch, prologue, and Online Softmax reduction overhead, whereas smaller intervals reduce head-of-line queuing delay under contention. Default chunk quanta can be selected from hardware tile multiples and **usable node VRAM relative to per-session KV size**: when a node holds only 1–2 ultra-long KV sessions, contention probability is low and larger quanta favor throughput; when usable VRAM holds many concurrent sessions, smaller quanta bound tail latency.

#### 3. Summary of Stage Sharing Asymmetry
1. **`Attn+KV` Stages:** Linear $W_{q,k,v,o}$ projections (`56–357 MiB` on `7B`–`70B`) can batch ready queries if concurrent arrivals coincide, whereas KV cache scans read private per-session state (`1.00 GiB` per `256K` tokens/layer) and rely on sequence chunking (**self-striping** and **concurrency-striping**).
2. **`FFN` Stages:** Hold zero KV state (`1.31 GiB` resident weights on Llama 70B; `2.62 GiB` on Mixtral 8x7B). When $B$ requests arrive within coalescing window $\Delta t_{\text{wait}}$, Dense FFN stages reuse resident weights across all $B$ tokens up to the Roofline crossover scale $B_{\text{crit}}$ (with MoE stages realizing lower per-expert reuse $B_{\text{eff}} \le B$ due to routing scatter).
3. **Independent Pool Elasticity:** Overloaded `FFN` stages can add stateless replicas without copying KV caches, while `Attn+KV` capacity scales horizontally with total active session tokens.

---

## 5. Analytical Hardware Performance & Cost Modeling

### 5.1 Idealized Single-Stream Roofline Walkthrough (`Llama 3.1 70B`, Unquantized 16-Bit)

To establish an idealized upper-bound baseline for an isolated request ($B = 1$, $S = 4{,}096$ tokens) before multi-user batching or intra-layer disaggregation, we account explicitly for all resident and active bytes of unquantized `Llama 3.1 70B`:
* **Resident Model Footprint:** $80\text{ layers} \times 1{,}632.0\text{ MiB} = \mathbf{127.50\text{ GiB}}$ (`136.90 GB` decimal), plus untied input embeddings and `lm_head` ($2 \times 128{,}256 \times 8{,}192 \times 2\text{ B} = \mathbf{3.91\text{ GiB}}$ / `4.20 GB`), totaling **`131.41 GiB`** (`141.10 GB`) of static weights before workspace or KV state (`+1.25 GiB` KV at $S = 4{,}096 \implies \mathbf{132.66\text{ GiB}}$ total).
* **Active Bytes & FLOPs per Decode Token ($B = 1, S = 4{,}096$):** Input embedding performs a single `16 KiB` row lookup, whereas all 80 layers (`127.50 GiB`), `lm_head` (`1.96 GiB`), and single-session KV cache ($80 \times 16.0\text{ MiB} = \mathbf{1.25\text{ GiB}}$) are read per step, totaling **`130.71 GiB` (`140.35 GB` decimal)** and **`149.74 GFLOPs` (`74.87 GMACs`)**.

Table 5.1 evaluates **homogeneous, capacity-fitting configurations** (accounting for software-usable VRAM from Table 3.2) and separates **Pipeline Parallelism (`PP`)**—where an isolated $B = 1$ token traverses stages sequentially at **single-chip** memory bandwidth, paying $P - 1$ point-to-point hop latencies—from **Tensor Parallelism (`TP`)**—where $K$ tightly coupled chips read $1 / K$ of every weight and GQA KV shard ($H_{\text{kv}} = 8$) simultaneously at **$K \times$ aggregate** memory bandwidth, paying $2L = 160$ `AllReduce` collectives per token over NVLink or TPU ICI:

$$\text{Time}_{\text{PP}}(B=1) = \sum_{s=1}^{P} \max\!\big(T_{\text{FLOP},s},\ T_{\text{VRAM},s}\big) + (P - 1) T_{\text{hop}}, \qquad \text{Time}_{\text{TP}=K}(B=1) = \max\!\left(\frac{T_{\text{FLOP}}}{K},\ \frac{T_{\text{VRAM}}}{K}\right) + 2L \cdot T_{\text{AllReduce}}$$

#### Table 5.1: Idealized Zero-Overhead Roofline Decode Bounds (`Llama 3.1 70B`, $B = 1, S = 4{,}096$)

| Accelerator SKU *(Usable Memory)* | Parallelism Mode & Capacity Fit (`132.66 GiB` Weights + KV) | Effective Memory Bus *(at $B=1$)* | Ideal Compute Time | Ideal VRAM Stream Time | Interconnect Overhead | **Ideal Time / Token** | **Ideal Upper-Bound TPS ($B=1$)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA T4** *(`~15.0 GiB`, FP16)* | **PP=10** *(8 layers/GPU, TCP)* | 320 GB/s *(1 GPU; 300+ rated)* | 2.30 ms | **438.6 ms** *(467.8 @ 300)* | +0.63 ms *(9 TCP hops)* | **439.2 ms** | **~2.3 tok/s** *(2.1 @ 300)* |
| **NVIDIA L4** *(`~22.5 GiB`)* | **PP=7** *(11–12 layers/GPU, TCP)* | 300 GB/s *(1 GPU active)* | 1.24 ms | **467.8 ms** | +0.42 ms *(6 TCP hops)* | **468.2 ms** | **~2.1 tok/s** |
| **Google TPU v4** *(`32.0 GiB`/chip)* | **PP=5** *(16 layers/chip, ICI)* | 1,200 GB/s *(1 chip active)* | 0.54 ms | **117.0 ms** | +0.01 ms *(4 ICI hops)* | **117.0 ms** | **~8.5 tok/s** |
| **NVIDIA A100 SXM** *(`~79.2 GiB`)* | **PP=2** *(40 layers/GPU, NVLink)* | 2,039 GB/s *(1 GPU active)* | 0.48 ms | **68.8 ms** | +0.002 ms *(1 NVLink)* | **68.8 ms** | **~14.5 tok/s** |
| **NVIDIA H100 SXM** *(`~79.2 GiB`)* | **PP=2** *(40 layers/GPU, NVLink)* | 3,350 GB/s *(1 GPU active)* | 0.15 ms | **41.9 ms** | +0.002 ms *(1 NVLink)* | **41.9 ms** | **~23.9 tok/s** |
| **NVIDIA H100 SXM** *(`~79.2 GiB`)* | **TP=2** *(Sharded across 2 GPUs, NVLink)* | 6,700 GB/s *(2 GPUs parallel)* | 0.08 ms | **20.9 ms** | +0.64 ms *(160 AllReduces)* | **21.6 ms** | **~46.3 tok/s** |
| **Google TPU `v4-16`** *(`256.0 GiB`)* | **TP=8** *(8 chips, 1 KV head/chip, ICI)* | 9,600 GB/s *(8 chips parallel)* | 0.07 ms | **14.6 ms** | +0.96 ms *(160 ICI syncs)* | **15.6 ms** | **~64.2 tok/s** |
| **Google TPU `v5p-8`** *(`380.0 GiB`)* | **TP=4** *(4 chips, 2 KV heads/chip, ICI)* | 11,060 GB/s *(4 chips parallel)* | 0.08 ms | **12.7 ms** | +0.64 ms *(160 ICI syncs)* | **13.3 ms** | **~75.0 tok/s** |
| **NVIDIA HGX B200** *(`≤180 GB`)* | **Single GPU** *(`142.4 GB` decimal fits 1 GPU)* | 8,000 GB/s *(1 GPU)* | 0.07 ms | **17.5 ms** | **0.00 ms** *(0 hops)* | **17.5 ms** | **~57.0 tok/s** |

*(Capacity Note on `TPU v4-8` vs. `v4-16` and `HGX B200`):* A 4-chip `TPU v4-8` host provides `128.0 GiB` of HBM, falling `3.41 GiB` short of `Llama 3.1 70B`'s `131.41 GiB` unquantized static weights even before workspace or KV state. Serving unquantized 70B on `v4-8` would require offloading embeddings and several layers to an auxiliary host—introducing stage compute, memory, and hop costs specific to that auxiliary hardware—so Table 5.1 reports the clean homogeneous `v4-16` (`8-chip`, `256.0 GiB`) slice instead. On `HGX B200`, `132.66 GiB` equals `142.44 GB` decimal, fitting within the `≤180 GB` (`~167.6 GiB`) software-usable HBM3e budget with `~25 GiB` to spare. All timings in Table 5.1 are **idealized Roofline lower bounds** (sustained decode bandwidth typically reaches $70\%\text{–}85\%$ of peak).

---

### 5.2 Illustrative Cloud Cost Structure ($B = 1$ Unbatched vs. High-Concurrency Regimes)

Using representative cloud on-demand and spot market rates (which vary across providers and regions), Table 5.2 illustrates why low-hourly-cost GPUs exhibit poor unbatched ($B = 1$) token economics unless stage batching ($B \gg 1$) amortizes weight reads:

#### Table 5.2: Indicative Cloud Rental Rates & Idealized Unbatched ($B = 1$) Cost per 1M Tokens (`Llama 70B`)

| Accelerator Config | Indicative On-Demand *(per Chip/GPU)* | Indicative Spot *(per Chip/GPU)* | Fleet Spot Rate | Unbatched ($B=1$) Cost / 1M Tokens *(Roofline Bound)* | **Structural Economic Driver** |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **NVIDIA T4** *(PP=10, FP16)* | ~\$0.55 / hr | ~\$0.15 / hr | **~\$1.50 / hr** | **~\$183–\$195** | **Low-Bandwidth Penalty at $B=1$:** 300–320 GB/s GDDR6 requires ~439–468 ms/token sequentially; competitive only when FFN queues sustain $B \gg 1$ weight reuse. |
| **NVIDIA L4** *(PP=7)* | ~\$0.84 / hr | ~\$0.28 / hr | **~\$1.96 / hr** | **~\$255** | 300 GB/s memory bus requires sustained stage batching ($B \ge 16\text{–}32$) to amortize weight streaming. |
| **NVIDIA A100 SXM** *(PP=2)* | ~\$3.67 / hr | ~\$1.35 / hr | **~\$2.70 / hr** | **~\$52** | 2.04 TB/s HBM2e reduces sequential weight-read time by $6.4\text{–}6.8\times$ vs. T4/L4. |
| **NVIDIA H100 SXM** *(PP=2 / TP=2)* | ~\$3.50–\$4.50 / hr | ~\$2.25 / hr | **~\$4.50 / hr** | **~\$52 (PP) / ~\$27 (TP)** | NVLink TP=2 doubles effective HBM bandwidth at $B=1$; high crossover ($B_{\text{crit}} \approx 295$) scales under batching. |
| **Google TPU `v4-16`** *(TP=8)* | \$25.76 / hr *(8 chips)* | ~\$9.00 / hr | **~\$9.00 / hr** | **~\$39** | 9.6 TB/s aggregate HBM bandwidth across 8 chips over optical ICI with clean GQA $H_{\text{kv}}=8$ sharding. |
| **Google TPU `v5p-8`** *(TP=4)* | ~\$16.80 / hr *(4 chips)* | ~\$8.40 / hr | **~\$8.40 / hr** | **~\$31** | Fits full 70B + generous KV headroom in `380 GiB` HBM2e at 11.1 TB/s aggregate bandwidth. |
| **NVIDIA HGX B200** *(Single GPU)* | ~\$5.50–\$6.50 / hr | ~\$6.00 / hr | **~\$6.00 / hr** | **~\$29** | Fits 70B + moderate KV on one GPU (`≤180 GB` usable) at 8 TB/s with zero network hops. |

---

## 6. Model-Derived Operating Regimes & Practitioner Hypotheses

Combining the intra-layer weight asymmetry (Section 2), wire serialization and hop latency equations (Section 3.1), weight-only Roofline crossover $B_{\text{crit}}$ (Section 3.3), and queuing separation between $M$ and $B$ (Section 4) yields four **testable operating hypotheses** that guide our experimental design and fleet-planning tool:

### 6.1 Candidate Operating Regimes (Hypotheses to Validate Empirically)

| Workload Regime | Dominant Constraint | Candidate Topology | Target Hardware Profile |
| :--- | :--- | :--- | :--- |
| **1. Low Concurrency, Short/Moderate Context** <br>($M \le 4$, $S \le 8\text{K}$) | Sequential weight-streaming time & per-hop network latency | **Consolidated Single-Host (`Single GPU` or `NVLink/ICI TP`) — Avoid TCP Disaggregation** | **1x HGX B200**, **2x H100 (NVLink TP)**, or **TPU `v5p-8` / `v4-16`** |
| **2. High Concurrency, Long Context** <br>($M \gg P$ yielding $B \ge 16$ at FFN, $S = 32\text{K–}256\text{K}$) | Multi-session KV VRAM capacity wall ($M \times \text{KV}_{\text{session}}$) & mixed-length contention | **Option C (`Attn+KV` Pool $\to$ Coalesced KV-Stateless `FFN` Stages)** | **Compute/HBM `FFN` Workers** + **Capacity-Scaled `Attn+KV` Session Nodes** |
| **3. Ultra-Long Context ($S \ge 256\text{K}$)** <br>(Private per-session KV scans; mixed short/long arrivals) | Per-session KV scan bandwidth & head-of-line SM monopolization | **Option C + Tuned Tier 2 Interruption Intervals** *(plus Tier 3 Online Softmax fan-out when $T_{\text{KV\_scan}} \gg T_{\text{TCP}}$)* | **Bandwidth-Optimized `Attn+KV` Nodes** (self- & concurrency-striping) + **Batched `FFN` Nodes** |
| **4. Latency-Tolerant Throughput / Legacy Fleet** <br>($M \gg P$ sustaining large $B$ toward $B_{\text{crit}}$) | Cost per million tokens ($\text{TPS} / \text{\$/hr}$) on constrained usable VRAM | **Unidirectional Pipeline with Coalescing Window $\Delta t_{\text{wait}}$ Building $B$** | **L4 / T4 / TPU `v5e` `FFN` Stages** paired with disaggregated `Attn+KV` nodes |

### 6.2 Practical Design Pitfalls & Open Heuristic Bounds

1. **Pitfall 1 — Running Unbatched ($B = 1$) Interactive Traffic on Low-Bandwidth Commodity Fleets:**
   On `300–320 GB/s` GPUs (T4/L4), an isolated $B = 1$ decode step operates far below $B_{\text{crit}} \approx 203\text{–}403$, yielding `~2.1–2.3 tok/s` on 70B and higher per-token cost than frontier HBM accelerators. Commodity GPU pipelines require sufficient global concurrency $M$ and stage coalescing to sustain $B \gg 1$ on shared FFN weights.
2. **Pitfall 2 — Pinning Large Multi-Session KV Pools on MoE/FFN Compute Silicon:**
   In MoE architectures such as Mixtral 8x7B (`97.1%` resident bytes in experts) and DeepSeek-V3 (`98.4%`), co-locating hundreds of gigabytes of session KV caches on the same accelerators holding resident expert pools restricts session capacity $M_{\text{node}}$ and ties up compute clusters during private KV scans.
3. **Pitfall 3 — Multi-Layer Synchronous TCP Ping-Pong:**
   Hosting many layers per worker while issuing synchronous round-trip TCP calls per layer compounds network hops ($2\times$ per layer) and mixes incompatible layer queues on return. Grouping local layers or partitioning into unidirectional `Attn+KV` $\to$ `FFN` stages separates stateful KV chunking from FFN batching.
4. **Spatial Cluster Ceiling When Packing Multiple Layers on Large Chips:**
   When spatially striping concurrent layer executions across a single large accelerator, the number of layers executing genuinely in parallel is bounded by the number of independent compute clusters (**SMs / TensorCores**—and the minimum SM group size needed to sustain memory-level parallelism) on the die. This bound is particularly tight on architectures with few coarse clusters (such as `2 TensorCores` per TPU v4/v5p chip or `8 TensorCores` per `-8` host), where exceeding the cluster count forces time-slicing rather than spatial concurrency.

---

## References

* **Borzunov, A., Baranchuk, D., Dettmers, T., Ryabinin, M., Belkada, Y., Chumachenko, A., Samygin, P., & Raffel, C. (2023).** *Petals: Collaborative Inference and Fine-Tuning of Large Models.* Proceedings of ACL 2023 (System Demonstrations), pp. 558–568.
* **Dao, T. (2023).** *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* arXiv preprint arXiv:2307.08691 (ICLR 2024).
* **Dao, T., Haziza, D., Massa, F., & Sizov, G. (2023).** *Flash-Decoding for Long-Context Inference.* Stanford CRFM / PyTorch Technical Blog, October 2023.
* **DeepSeek-AI (2026).** *DeepInfer: High-performance, large-scale inference framework for DeepSeek models.* GitHub repository, `https://github.com/deepseek-ai/DeepInfer`.
* **Google Cloud (2024a).** *Cloud TPU v4 System Architecture.* Google Cloud Documentation, `https://cloud.google.com/tpu/docs/v4`.
* **Google Cloud (2024b).** *Cloud TPU v5p System Architecture.* Google Cloud Documentation, `https://cloud.google.com/tpu/docs/v5p`.
* **Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., & Stoica, I. (2023).** *Efficient Memory Management for Large Language Model Serving with PagedAttention.* Proceedings of ACM SOSP 2023, pp. 611–626.
* **NVIDIA Corporation (2024a).** *NVIDIA Collective Communications Library (NCCL) User Guide: Point-to-Point Communication.* `https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html`.
* **NVIDIA Corporation (2024b).** *NVIDIA T4, L4, A100, and H100 Tensor Core GPU Architecture Whitepapers and Datasheets.* NVIDIA Technical Documentation.
* **NVIDIA Corporation (2025).** *NVIDIA Blackwell Tuning Guide: High-Bandwidth Memory (HBM3/HBM3e) Subsystem.* CUDA Toolkit Documentation, `https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html`.
* **Patel, P., Choukse, E., Zhang, C., Shah, A., Goiri, Í., Maleki, S., & Bianchini, R. (2024).** *Splitwise: Efficient Generative LLM Inference Using Phase Splitting.* Proceedings of ACM/IEEE ISCA 2024, pp. 118–132.
* **Qin, R., Li, Z., He, W., Zhang, M., Wu, Y., Zheng, W., & Xu, X. (2024).** *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.* arXiv preprint arXiv:2407.00079 (FAST 2025).
* **Rajbhandari, S., Li, C., Yao, Z., Zhang, M., Aminabadi, R. Y., Awan, A. A., Rasley, J., & He, Y. (2022).** *DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale.* Proceedings of ICML 2022 (PMLR 162), pp. 18332–18346.
* **Song, Y., Mi, Z., Xie, H., & Chen, H. (2024).** *PowerInfer: Fast Large Language Model Serving with a Consumer-Grade GPU.* Proceedings of ACM SOSP 2024, pp. 590–606.
* **vLLM Project & StepFun (2026).** *vLLM Attention-FFN Disaggregation Plugin (`afd-plugin`) and StepMesh RDMA Communication Engine.* GitHub repository, `https://github.com/vllm-project/afd-plugin`.
* **Yu, G.-I., Jeong, J. S., Kim, G.-W., Kim, S., & Chun, B.-G. (2022).** *Orca: A Distributed Serving System for Transformer-Based Generative Models.* Proceedings of USENIX OSDI 2022, pp. 521–538.
* **Zhong, Y., Liu, S., Chen, J., Hu, J., Zhu, Y., Liu, X., Jin, X., & Zhang, H. (2024).** *DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving.* Proceedings of USENIX OSDI 2024, pp. 193–210.

---

# TBD
