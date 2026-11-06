# FrontierSplit High-Bandwidth Flow Architecture (`TPU` & `HBM` `GPU`)

This document specifies the hardware trade-offs, analytical Roofline estimates, execution hypotheses, and correctness contracts for FrontierSplit's multi-stage flow architecture prior to implementing the stage modules, runners, and scheduler.

---

## 1. Architectural Scope, Hardware Regimes, & Acceptance Criteria

### 1.1 Two Hardware Regimes in FrontierSplit

1. **Primary Repository Focus — Fast, High-Bandwidth `TPU` / `GPU` Clusters *(This Document)*:**
   - Targets high-bandwidth `HBM` accelerators (**`Google Cloud TPU v4` / `v5p` slices** first, with direct portability to **`NVIDIA A100` / `H100`** GPUs), where per-layer memory streaming (`8–91 µs` on `Mistral-7B`) is faster than Host OS socket and PCIe setup (`~100–300 µs`).
   - Co-locates **`L_stage` contiguous full layers (`Attn + KV + FFN`)** per stage host (sharded intra-host across chips via `ICI` / `NVLink`), processes fixed-capacity container waves up to **`256K` KV tokens**, and paces staggered waves across `P` pipeline stages.
2. **Prior / Low-Bandwidth & Local Hardware Scope *([`paper/frontiersplit_technical_paper_draft2.md`](paper/frontiersplit_technical_paper_draft2.md))*:**
   - On lower-bandwidth commodity GPUs (`300 GB/s` `NVIDIA T4` / `L4`) or constrained local/non-cloud hosts where a single layer takes `1.5–9.7 ms` (`5×–50×` longer than the OS/PCIe bridge), intra-layer `Attn+KV ↔ FFN` disaggregation (Option C) remains viable. That design is preserved as prior analytical work and candidate future experiments rather than discarded, while the present repository focuses on the high-bandwidth `L_stage`-layer flow design below.

### 1.2 Role of Reference Baselines & Validation Criteria

- **Single-Machine Consolidated Reference (`hugging_face/` and `baseline/`):**
  - Intended strictly for single-machine execution on top-tier consolidated hardware (e.g., `HGX B200` / `H200` where unquantized `BF16` weights and active KV caches fit in local `HBM` with zero network hops), serving as the consolidated reference benchmark for `Mistral-7B-Instruct-v0.3` (`13.50 GiB` weights) and `Mixtral-8x7B-Instruct-v0.1` (`86.99 GiB` resident weights; `23.74 GiB` active at `B = 1` when 2 experts fire).
- **Numerical & Evaluation Acceptance Criteria:**
  - Bitwise equality (`rtol=0, atol=0`) applies specifically between `baseline/` and `hugging_face/` on the same device and kernel path.
  - Across chunked `FP32` Online Softmax reductions, static batch containers, 4-chip sharding, and `TPU` vs. `GPU` hardware (where floating-point reduction order legitimately differs), correctness is judged by bounded numerical tolerances against `baseline/` on stage/end-to-end outputs alongside downstream benchmark parity per [`evals.md`](evals.md).
- **Container Capacity (`256K`) vs. Checkpoint Context Training (`32K`):**
  - Both pinned baseline checkpoints specify `max_position_embeddings = 32,768`. Supporting backend KV containers up to `256K` tokens validates systems memory/scheduling mechanics and is distinct from claiming model quality beyond `32K` without long-context checkpoints or RoPE scaling.
- **Portable Math vs. Compiler Execution Hypotheses:**
  - Stage modules use standard static-shape PyTorch tensor operations parameterized by fixed-shape runtime metadata tensors (block indices, session ownership, valid lengths, and causal positions) whose *values* change every step without triggering XLA graph recompilation.
  - Hardware execution properties discussed below (`VMEM` residency, async DMA overlap, achieved `HBM` utilization) are compiler/runtime targets to verify empirically under `torch_xla` / PJRT (and potentially refine via custom kernels where compiler scheduling falls short).

### 1.3 Notation

| Symbol | Meaning |
| :---: | :--- |
| $L$ | Total transformer layers in the model ($L = 32$ for `Mistral-7B` and `Mixtral-8x7B`). |
| $P,\ L_{\text{stage}}$ | Number of pipeline stages ($P$) and contiguous full layers per stage ($L_{\text{stage}} = L / P$). |
| $M,\ M_{\text{resident}}$ | Global concurrent requests in flight across the `P`-stage pipeline ($M$) vs. total sessions holding resident KV blocks on a stage ($M_{\text{resident}} \ge M$). |
| $B,\ N_{\text{prefill}}$ | Active `1`-token decode requests in a wave ($B$) and piggybacked prefill tokens in that wave ($N_{\text{prefill}}$), bounded by $B + N_{\text{prefill}} \le B_{\text{max}} = 128$. |
| $S_{\text{bucket}},\ s_{\text{valid}}$ | Session's allocated KV capacity bucket ($S_{\text{bucket}}$, a multiple of `128` up to `256K`) vs. currently populated valid token count ($s_{\text{valid}} \le S_{\text{bucket}}$). |
| $K_{\text{wave}},\ K_{\text{pool}}$ | Single-wave KV scan container budget ($K_{\text{wave}} = 256\text{K}$ tokens = `256` blocks of `1,024`) vs. total stage `HBM` KV pool capacity ($K_{\text{pool}} \gg K_{\text{wave}}$). |

---

## 2. TPU Hardware Anatomy, 4-Chip Sharding, & Communication Cliffs

### 2.1 Silicon Hierarchy (`TPU v4-8` & `TPU v5p-8`)

- **Host & Chip Layout:** `1` Linux Host VM connects over `4` independent PCIe links to `4` TPU chips (`v4-8`: `128 GiB` `HBM` at `4.8 TB/s` peak aggregate; `v5p-8`: `380 GiB` `HBM` at `11.06 TB/s` peak aggregate). Each `v4` / `v5p` chip contains `2` physical TensorCores (`8` TensorCores per `-8` host), exposed by default in **MegaCore** mode as `4` logical chip devices (`8` MXUs per chip).
- **Inside `1` TensorCore (`1` Instruction Stream, `3` Overlapped Engines):**
  - Unlike an NVIDIA GPU (`108–132` independent SMs), a TPU TensorCore executes **one** VLIW instruction stream, overlapping three functional engines across loop iterations when scheduled cleanly by the compiler:
    1. **DMA Engine:** Moves blocks asynchronously between off-chip `HBM` and on-chip `VMEM` scratchpad SRAM (`16–32+ MiB`).
    2. **`4` `[128 × 128]` Systolic MXUs:** Execute `BF16` matrix multiplies (`16,384` MACs per MXU = `64` `16 × 16` GPU tiles) with `FP32` accumulation.
    3. **Vector Processing Unit (`VPU`):** Executes `FP32` vector and reduction ops (`RMSNorm`, `RoPE`, `SiLU`, masking, and Online Softmax `max`/`exp`/rescaling) in `VMEM` and vector registers (`VREGs`).

### 2.2 Space (`1 MXU`) × Time (`128` Cycles) Sliding Window

- A matrix multiply consumes **only `1` MXU** (leaving the other `3` MXUs available for other `[128, 128]` weight tiles), while input vectors stream out of `VREGs` / `VMEM`.
- Pre-loading the next `[128, 128]` weight tile into an MXU's shadow latches takes **`128` clock cycles**, during which up to **`128` input vectors (`[1, 128]`)** can enter at `1` vector/cycle. Passing `B = 1` leaves `127` of those `128` systolic cycles unused without wasting `HBM`/`VMEM` storage; adding batch rows up toward `128` fills otherwise-idle compute cycles with minimal extra wall-clock cost while `HBM`-bandwidth bound ($B_{\text{crit}} \approx 166\text{–}229$).

### 2.3 Contiguous `HBM → VMEM` Block Sizing

- Base math tiles are **`128 × 128` (`32 KiB` in `BF16`)**, including zero-padded ragged tails.
- Scattering individual `32 KiB` tiles randomly in `HBM` degrades TPU DMA throughput due to per-jump descriptor overhead and `HBM` row misses. Grouping `8` tiles into **contiguous `1,024`-token blocks** (**`256 KiB` for `K` and `256 KiB` for `V` = `512 KiB` combined per KV head**, or `4.00 MiB` across `8` KV heads per layer) targets sequential burst DMA bandwidth while fitting `VMEM` ping-pong buffers.

### 2.4 Explicit 4-Chip Intra-Stage Sharding & Communication Cliffs

To utilize all `4` chips (`4.8–11.1 TB/s` aggregate `HBM`) from **`1` whole-host SPMD worker process per `-8` VM** with minimal collective traffic (`2` `AllReduce`s per layer = `64` across `32` layers):
1. **`Attention` (Head-Parallel):** Shard `W_q, W_k, W_v` column-wise by GQA groups (**`8` Q heads + `2` KV heads per chip** across the `4` chips). Each chip stores and scans only its `2` KV heads locally (**`0` `ICI` traffic during KV attention**), followed by row-parallel `W_o` and **`1` `ICI` `AllReduce`** (`[B_total, 4096]`).
2. **Dense `FFN` (`Mistral-7B`):** Column-parallel `gate_proj` / `up_proj` (`d_ffn / 4 = 3,584 = 28 × 128` per chip) + row-parallel `down_proj` + **`1` `ICI` `AllReduce`**.
3. **MoE `FFN` (`Mixtral-8x7B`):** Replicate the `64 KiB` router and shard **each expert's** `d_ffn = 14,336` 4-way across all `4` chips (`3,584` per chip) + **`1` `ICI` `AllReduce`**. Unlike whole-expert-per-chip placement (which leaves chips idle when `B = 1` activates only `2` experts), 4-way intra-expert sharding drives all `4` chips in parallel regardless of which experts fire.

| Boundary | Physical Path | Est. Latency (`8–16 KiB`) | Bandwidth | Execution Impact |
| :--- | :--- | :---: | :---: | :--- |
| **1. Intra-Core Chunks** | Sequential loop in `VMEM` / `VPU` | **`~0 µs`** *(target)* | Multi-`TB/s` SRAM | Running `(A, m, l)` stays on-chip across sequential blocks when fused. |
| **2. Intra-Slice (`ICI`)** | Inter-Chip Interconnect (`v4-8`, `v4-16`, `v5p-8`) | **`~1 – 4 µs`** | `50–100 GB/s`/link | Inside XLA via `ICI` routers/DMA (copper intra-tray, optical inter-tray in pods); zero PCIe/CPU work. |
| **3. Cross-VM (`TCP`)** | `TPU HBM → PCIe → Host RAM → VPC TCP → Host RAM → PCIe → TPU HBM` | **`~150 – 250 µs`** *(est.)* | `6–25 GB/s` | Exits XLA, crosses PCIe twice + POSIX TCP, and consumes Host CPU + PJRT dispatch budget. |

---

## 3. Idealized Roofline Cycle Bounds, Host Ingest, & Stage Granularity

### 3.1 Idealized Weight-Only `HBM` Lower Bounds (Zero KV Cache, Zero Network Hops)

All figures below are **idealized peak-bandwidth lower bounds** (`Decimal Weight Bytes ÷ Peak Aggregate HBM Bandwidth`: `1.2 MB/µs` on `1× v4`, `4.8 MB/µs` on `4× v4`, `11.06 MB/µs` on `4× v5p`), prior to kernel/collective overhead:

#### Per Single Layer (`1` of `32` Layers)
| Weight Block (`BF16`) | Binary (`MiB`) | Decimal (`MB`) | `1× TPU v4` (`1.2 MB/µs`) | `4× TPU v4` (`4.8 MB/µs`) | `4× TPU v5p` (`11.1 MB/µs`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Attn` Only** (`W_q,k,v,o`) | `80 MiB` | `83.9 MB` | `70 µs` | **`17.5 µs`** | **`7.6 µs`** |
| **Mistral Dense `FFN`** *(`B ≤ 128`)* | `336 MiB` | `352.3 MB` | `294 µs` | **`73.4 µs`** | **`31.9 µs`** |
| **Mistral Full Layer** (`Attn + FFN`) | `416 MiB` | `436.2 MB` | `364 µs` | **`90.9 µs`** | **`39.5 µs`** |
| **Mixtral `FFN`** *(`2` experts active, e.g. `B = 1`)* | `672 MiB` | `704.6 MB` | `587 µs` | **`146.8 µs`** | **`63.7 µs`** |
| **Mixtral `FFN`** *(all `8` experts active)* | `2,688 MiB` | `2,819 MB` | *(2,349 µs)* | **`587.2 µs`** | **`254.9 µs`** |

#### Across All `32` Layers (Including `lm_head` `~268 MB`)
| Full `32`-Layer Weight Pass | Total Weights | `1× TPU v4` | `4× TPU v4` (`v4-8`) | `4× TPU v5p` (`v5p-8`) |
| :--- | :---: | :---: | :---: | :---: |
| **`Attn` Only (`32` layers)** | `2.50 GiB` (`2.68 GB`) | `2.24 ms` | **`0.56 ms`** | **`0.24 ms`** |
| **Mistral `FFN` Only (`32` layers)** | `10.50 GiB` (`11.27 GB`) | `9.40 ms` | **`2.35 ms`** | **`1.02 ms`** |
| **Mistral Full Model** (`Attn + FFN + lm_head`) | `13.25 GiB` (`14.23 GB`) | `11.86 ms` | **`2.96 ms`** | **`1.29 ms`** |
| **Mixtral Full Model** *(`2` experts/layer active)* | `23.74 GiB` (`25.49 GB`) | *(exceeds 1 chip)* | **`5.31 ms`** | **`2.30 ms`** |
| **Mixtral Full Model** *(all `8` experts/layer active)* | `86.74 GiB` (`93.14 GB`) | *(exceeds 1 chip)* | **`19.40 ms`** *(whole wave)* | **`8.42 ms`** *(whole wave)* |

*(Estimated interconnect comparison: `64` intra-host `ICI` `AllReduce`s add `~0.15–0.25 ms` total per token, whereas `64` cross-VM `PCIe + TCP` hops add `~9.6–16.0 ms` per token.)*

### 3.2 Working Host-Ingest Estimates Expressed in Equivalent TPU Layers

Batching `B` activations into one application frame still segments into multiple MTU packets on the NIC, but reduces Python/`asyncio` framing calls and PJRT device submissions from `B` to `1`. Using working estimates of `~5–10 µs` per application socket read/write, `~40–60 µs` per unilateral `H2D` buffer transfer, and `~80–150 µs` for a complete `H2D + Execute + D2H` step submission *(all to be measured directly on target TPU VMs)*:

| Stage Ingest Strategy | Est. Serial Host + Submission Time | Equivalent **Mistral** Layers (`v4-8`) | Equivalent **Mistral** Layers (`v5p-8`) | Equivalent **Mixtral** Layers (`v4-8`) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Per-request socket read + TPU `H2D` (`10` reqs)** | **`~450 – 700 µs`** | **`5 – 8` layers** | **`11 – 18` layers** | `0.8 – 4` layers |
| **2. `16` individual app frames → Host RAM → `1` TPU step** | **`~240 – 350 µs`** | **`2.6 – 4` layers** | **`6 – 9` layers** | `0.4 – 2` layers |
| **3. `1` batched `[B_total, 4096]` app frame → `1` TPU step** | **`~100 – 170 µs`** | **`1.1 – 1.9` layers** | **`2.5 – 4` layers** | **`0.2 – 1` layer** |

### 3.3 Hardware Inversion & Coarse `L_stage`-Layer Co-Located Stages

- **Hardware Inversion:** On `300 GB/s` `T4`/`L4`, `1` layer (`1.45–9.68 ms`) easily hides `~150–300 µs` of host/socket overhead. On `4.8–11.1 TB/s` TPUs, `1` layer (`40–91 µs` on Mistral; `8–18 µs` on `Attn` alone) finishes faster than the Host OS/PCIe submission path, starving `1`-layer or `64`-hop per-layer TCP splits.
- **Coarse `L_stage`-Layer Stages (`Attn + KV + FFN` Together):**
  - Hosting **`L_stage` contiguous full layers** (`e.g., L_stage = 4, 8, 16, or 32`) plus those layers' resident KV pool on each stage VM gives $T_{\text{work}}(L_{\text{stage}}) \gg T_{\text{net}}$ (`~0.5–3.3 ms` network breathing room vs. `~0.15 ms` wave transfer) and `0–8` cross-VM hops/token.
  - Because `L_stage` weights occupy a modest share of `128–380 GiB` `HBM` (`10.5%` for `32` Mistral layers on `v4-8`; `17%` for `8` Mixtral layers on `v4-8`), scaling session KV capacity horizontally adds stage replicas with duplicated `L_stage` weights ("single-hop per slice").
  - Staggered waves circulate on the wire (double-buffered into shadow `HBM` buffers during active execution) with **work-conserving dispatch** (`0 µs` artificial wait when idle).

---

## 4. Bucketed KV Containers, Online Softmax, & Flow Scheduling

### 4.1 Equal-Byte Wave Budget Heuristic ($B \times S_{\text{bucket}} \le K_{\text{wave}}$)

Private session KV reads (`4.00 KiB / tok / layer`) scale linearly with active sessions (`7.0 µs`, `28.0 µs`, and `111.8 µs` idealized `HBM` lower bound per session/layer at `8K`, `32K`, and `128K` on `v4-8`).

Assigning sessions (input + capped output budget) to **power-of-2 `128`-multiple buckets ($S_{\text{bucket}}$)** and capping total wave KV bytes at **$K_{\text{wave}} = 256\text{K}$ tokens (`1.00 GiB/layer`)** equalizes idealized `HBM` bytes across bucket tiers:

| Bucket Size ($S_{\text{bucket}}$) | Decode Batch Cap ($B$) | Wave KV Budget ($B \times S_{\text{bucket}}$) | Ideal KV `HBM` Bound / Layer | **Ideal Mistral Layer Bound** (`+91 µs`) | **Ideal Mixtral Layer Bound** (`+605 µs` @ 8 exp) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`2,048` (`2K`)** | **`128`** | `256K` tok (`1.00 GiB`) | **`224 µs`** | **`315 µs`** | **`829 µs`** |
| **`8,192` (`8K`)** | **`32`** | `256K` tok (`1.00 GiB`) | **`224 µs`** | **`315 µs`** | **`829 µs`** |
| **`32,768` (`32K`)** | **`8`** | `256K` tok (`1.00 GiB`) | **`224 µs`** | **`315 µs`** | **`829 µs`** |
| **`131,072` (`128K`)** | **`2`** | `256K` tok (`1.00 GiB`) | **`224 µs`** | **`315 µs`** | **`≤ 829 µs`** |
| **`262,144` (`256K`)** | **`1`** | `256K` tok (`1.00 GiB`) | **`224 µs`** | **`315 µs`** | **`≤ 829 µs`** |

*(Modeling note: `224 µs` is an idealized `HBM` transfer lower bound; query gather, causal masking, `VPU` softmax/reduction, Mixtral expert count variance when $B < 8$, and mixed prefill introduce runtime differences that we treat as empirical quantities to measure rather than exact equal-duration guarantees.)*

### 4.2 Mixed-Bucket Knapsack Containers & Sequential-Per-Core Block Execution

- Dividing each stage's `HBM` KV pool ($K_{\text{pool}}$) into uniform **`1,024`-token blocks** allows **mixed bucket sizes inside one static XLA wave** whenever total active blocks $\le 256$ (`256K` tokens) and $B \le 128$ (e.g., `1×128K + 2×32K + 8×8K`).
- Physically, each chip shards across its `2` KV heads while iterating assigned `1,024`-token blocks sequentially through `VMEM`, producing partial `FP32` Online Softmax states `(A_p, m_p, l_p)` that segment-reduce by session index into `[B_max, H_q, 128]`.
- Pre-compiling `3–4` static container sizes (`e.g., 8K, 32K, 128K, 256K` KV budgets) avoids scanning empty blocks at low concurrency without runtime recompilation. For Mixtral, sorting `2 × B_total` routed token-expert pairs into static `ragged_dot` / `gmm` keeps XLA shapes fixed.

### 4.3 Completion-Paced Flow Scheduler

Because waves self-clock on stage completion (tolerating real step-time variance across waves):
1. **Stagger First ($M \le P$):** Places active requests into distinct circulating waves around the `P` stages so sessions sharing the same pinned machines avoid simultaneous contention where empty wave slots exist.
2. **Balanced Knapsack Packing ($M > P$):** Pins new sessions to stage replicas with `K_pool` headroom and assigns them to waves with the lightest estimated step load ($\sum s_{\text{valid}}$ + prefill FLOPs) to reduce convoy queuing behind heavy waves.

---

## 5. Piggybacked Chunked Prefill (`B + N_prefill ≤ 128`)

- Packing **`N_prefill ≤ 128 - B` prefill tokens** from an incoming prompt into unused rows of `[128, 4096]` shares `W_qkv`, `W_o`, and Dense `FFN` weight reads inside the MXU's `128`-cycle window (on Mixtral when $B < 8$, prefill tokens may activate additional experts beyond those selected by decode tokens alone).
- **Two-Dimensional Prefill Admission Bound (Rows *and* Attention FLOPs):**
  - While weight projections stay memory-bound for $B + N_{\text{prefill}} \le 128$, **prefill attention over a long existing prefix crosses into compute-bound execution** because GQA evaluates `4` Q heads per KV head (`e.g., 126` prefill queries over `128K` cached tokens = `~271 GFLOPs`/layer $\approx 246\ \mu\text{s}$ at `v4-8` peak `1,100 TFLOPS`, exceeding the `112 µs` KV `HBM` read bound).
  - Prefill admission per wave is therefore bounded both by spare token rows ($128 - B$) and by an attention-FLOP budget ($N_{\text{prefill}} \times S_{\text{prefix}}$), throttling $N_{\text{prefill}}$ on long-prefix turns so wave duration stays bounded (hypothesis to calibrate empirically).
- **Causal-Mask Chunk Pipelining:** Because causal masking prevents early chunk tokens (`0..99`) from attending to future prompt tokens (`100..299`), `Stage 0`'s output on Chunk A is immediately exact for `Stage 1`. Consecutive prompt chunks (`A, B, C, ...`) can therefore advance on consecutive heartbeats without waiting for a `P`-stage round trip.

---

## 6. Multi-Turn Tiered KV Caching (`Host CPU RAM ↔ Device HBM`)

To avoid recomputing multi-turn prompts (`~700–2,600 ms` at `32K`), idle session KV blocks offload to pinned Host CPU DRAM between turns and reload when the user replies.

### 6.1 Idealized PCIe Transfer Bounds (`~25 GB/s` Gen4 / `~55 GB/s` Gen5 per Chip Link)

Across `4` parallel chip PCIe links (`~100 GB/s` Gen4, `~220 GB/s` Gen5), an `L_stage`-layer host transfers only **`L_stage / 32`** of the session's KV state:

| Cached History ($S$) | Full `32`-Layer KV *(1-Box Bound)* | `1× Gen4 GPU` (`25 GB/s`, 32L) | `1× 4-Chip Host` (`100 GB/s` Gen4, 32L) | **Per `8`-Layer Stage Host** (`100 GB/s` Gen4 / `220 GB/s` Gen5) |
| :--- | :---: | :---: | :---: | :---: |
| **`2,048` (`2K`)** | `256 MiB` (`268 MB`) | `10.7 ms` | `2.7 ms` | **`0.67 ms` / `0.31 ms`** |
| **`8,192` (`8K`)** | `1.00 GiB` (`1.07 GB`) | `43.0 ms` | `10.7 ms` | **`2.7 ms` / `1.2 ms`** |
| **`32,768` (`32K`)** | `4.00 GiB` (`4.30 GB`) | `171.8 ms` | `43.0 ms` | **`10.7 ms` / `4.9 ms`** |
| **`131,072` (`128K`)** | `16.00 GiB` (`17.18 GB`) | `687.2 ms` | `171.8 ms` | **`43.0 ms` / `19.5 ms`** |

### 6.2 Contention Mitigation & Runtime Verification

- **Bandwidth Ratio:** `25–55 GB/s` PCIe is `~1.2%–2.1%` of `1,200–3,350 GB/s` per-chip `HBM` bandwidth, and hardware DMA engines do not consume `SM` / `MXU` arithmetic units.
- **PCIe & Socket Isolation (Mechanism Depends on Backend):**
  - Unchunked gigabyte transfers would head-of-line block active `[B_total, 4096]` wave activations sharing PCIe or NIC queues.
  - Swaps therefore operate at **`1,024`-token block granularity** (`~320 µs` per 8-layer slice across 4 Gen4 links), using separate TCP connections for any remote/storage → Host RAM KV reads. On CUDA GPUs, transfers use `pin_memory=True` on a separate low-priority copy stream; on `TPU` / PJRT (which does not expose CUDA stream priorities), overlap vs. between-beat scheduling of async `H2D`/`D2H` block transfers must be verified empirically in Stage 1.

---

## 7. Stage-1 Correctness Contracts & Initial Validation Plan

Before building the distributed scheduler, the first implementation target is a **single `L_stage`-layer stage** with explicit 4-chip sharding validated against `baseline/` across ordinary decode, chunked prefill, and mixed decode + piggybacked prefill under five invariants:

1. **Allocated Bucket ($S_{\text{bucket}}$) vs. Valid Prefix ($s_{\text{valid}}$):** Attention masks strictly enforce `key_pos < s_valid` and `key_pos <= query_pos` so unpopulated bucket capacity never contributes to outputs.
2. **Exact `FP32` Online Softmax Contract `(A_p, m_p, l_p)`:**
   - Per block `p` and query head: row maximum $m_p$, unnormalized sum $\ell_p = \sum \exp(\text{score} - m_p)$, and unnormalized value accumulator $A_p = \exp(\text{score} - m_p) V_p$ (final normalized output $O = A / \ell$).
   - Associative merge of `(A_1, m_1, l_1)` and `(A_2, m_2, l_2)`: $m = \max(m_1, m_2)$, $\ell = e^{m_1 - m}\ell_1 + e^{m_2 - m}\ell_2$, $A = e^{m_1 - m}A_1 + e^{m_2 - m}A_2$.
   - **Fully masked / unused block identity:** `(m_p = -inf, l_p = 0.0, A_p = 0.0)`, guarded via `torch.where` so `exp(-inf - (-inf))` never produces `NaN`.
3. **Explicit Query-to-Block & Causal-Limit Metadata:** Block metadata identifies owning session slots, while query metadata maps both `1`-token decode rows and multi-token prefill slices (`N_prefill`) to their session's KV blocks with per-query causal position cutoffs.
4. **KV Write-Completion & Buffer Ownership:** Scatter writes of newly projected `(K, V)` tokens into a stage's `HBM` KV pool complete before Attention reads that block in the current beat, and blocks undergoing `Host RAM ↔ HBM` offload/reload remain fenced out of wave admission until transfer completion.
5. **Wave Container ($K_{\text{wave}}$) vs. Resident Pool ($K_{\text{pool}}$):** `K_wave = 256` blocks bounds a single wave's gather/scan input, whereas `K_pool` holds resident blocks across all `P` circulating waves plus retained multi-turn sessions awaiting Host-RAM offload.

---

## 8. Open Questions & Parked Future Work

1. **Multiple Simultaneous Prefills, Load Balancing, & Scaling *(Scheduler Phase)*:** Admission fairness across concurrent incoming prompts (round-robin vs. shortest-prompt-first allocation of `N_prefill` budget, dedicated prefill waves when decode load is sparse, and replica placement).
2. **Ultra-Long Contexts (`1M – 10M` Token KV Caches) *(Parked Future Experiment)*:** Cross-host `Tier 3` Online Softmax KV fan-out when single-session KV exceeds single-stage capacity/bandwidth.
3. **Large / Unbounded Output Generation *(Parked Future Experiment)*:** Bucket promotion or migration when output tokens exceed a session's initial `S_bucket` cap.
4. **Low-Bandwidth / Local Hardware Disaggregation *(Parked / Prior Work)*:** Evaluating intra-layer `Attn+KV ↔ FFN` splits from [`paper/frontiersplit_technical_paper_draft2.md`](paper/frontiersplit_technical_paper_draft2.md) on `300 GB/s` GPUs (`T4`/`L4`) or heterogeneous local nodes.
