# FrontierSplit: Architecture Specification

## 1. System Overview

**FrontierSplit** is a dual-paradigm distributed inference architecture designed to break the memory and compute bottlenecks of large language models on commodity hardware clusters.

The repository isolates two distinct architectural approaches:

```
                                  ┌───────────────────────────────┐
                                  │   FrontierSplit Gateway       │
                                  │   (OpenAI REST API /v1/...)   │
                                  └──────────────┬────────────────┘
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
    ┌─────────────────────────────┐                             ┌─────────────────────────────┐
    │  frontiersplit/context/     │                             │  frontiersplit/pipeline/    │
    │  Disaggregated Context      │                             │  1F1B Pipeline Parallel     │
    │  Engine (Online Softmax)    │                             │  Slicing (Model Slicing)    │
    ├─────────────────────────────┤                             ├─────────────────────────────┤
    │ • Weightless Context Node   │                             │ • Layer-by-layer slicing    │
    │ • 8 KB Online Softmax RPCs  │                             │ • 1F1B micro-batch schedule │
    │ • Zero raw KV cache motion  │                             │ • 16 KB activation streaming│
    │ • Disagg. Delta Prefill     │                             │ • Worker-level local cache  │
    └─────────────────────────────┘                             └─────────────────────────────┘
```

1. **Disaggregated Context Engine ([`frontiersplit/context/`](file:///Users/pixel/code/FrontierSplit/frontiersplit/context/))**:
   - **Target**: Overcoming the "Long-Context VRAM Wall" (32k–128k token prompts).
   - **Mechanism**: The static prompt KV cache is offloaded to a dedicated "weightless" Context Server ($0$ model weights). The compute decode worker computes only local output tokens and issues tiny **8 KB Query vector RPCs** over TCP to the Context Server. The Context Server executes partial attention ($S = Q K^T, \text{softmax} \cdot V$) and returns partial accumulators $(A, m, \ell)$. The decode worker merges prompt and local attention using **Online Softmax rescaling** with zero mathematical approximation.
   - **Delta Prefill**: Multi-turn agent turns extend context by sending only suffix query vectors and committing delta KV caches over TCP without moving historical prompt tensors.

2. **1F1B Pipeline Parallel Engine ([`frontiersplit/pipeline/`](file:///Users/pixel/code/FrontierSplit/frontiersplit/pipeline/))**:
   - **Target**: Serving unquantized large models (e.g. Mixtral 8x7B FP16 ~94 GB) across nodes with small VRAM (e.g. 4x or 8x 16GB/24GB GPUs or Cloud TPUs).
   - **Mechanism**: The model layers are sliced vertically across stages. Nodes exchange only intermediate activation vectors ($\sim 16\text{ KB}$ per token) over persistent binary TCP. Pipeline bubbles are neutralized via 1F1B micro-batch scheduling fed by concurrent multi-agent streams.

3. **Common Ingress Gateway ([`frontiersplit/gateway.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/gateway.py))**:
   - Exposes an OpenAI-compatible HTTP REST and Server-Sent Events (SSE) streaming API (`/v1/chat/completions`, `/health`, `/v1/models`).
   - Supports routing incoming completions to either the Disaggregated Context backend or the Pipeline Parallel backend.

---

## 2. Design-to-Implementation Architecture Map

For detailed design specifications, mathematical proofs, and cloud scaling topologies, see [`docs/`](file:///Users/pixel/code/FrontierSplit/docs/README.md):

| Technique | Component | Design Specification | Implementation File | Verification Tests |
| :--- | :--- | :--- | :--- | :--- |
| **Context** | Online Softmax Kernel | [`docs/context/DESIGN_CLOUD_ARCHITECTURE.md`](file:///Users/pixel/code/FrontierSplit/docs/context/DESIGN_CLOUD_ARCHITECTURE.md#21-online-softmax-attention-math) | [`frontiersplit/context/online_softmax.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/context/online_softmax.py) | [`tests/test_online_softmax.py`](file:///Users/pixel/code/FrontierSplit/tests/test_online_softmax.py) |
| **Context** | Weightless Context Server | [`docs/context/DESIGN_CLOUD_ARCHITECTURE.md`](file:///Users/pixel/code/FrontierSplit/docs/context/DESIGN_CLOUD_ARCHITECTURE.md#22-disaggregated-prompt-serving--the-weightless-context-machine) | [`frontiersplit/context/context_server.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/context/context_server.py) | [`tests/test_context_server.py`](file:///Users/pixel/code/FrontierSplit/tests/test_context_server.py) |
| **Context** | Disaggregated Attention Worker | [`docs/context/DESIGN_CLOUD_ARCHITECTURE.md`](file:///Users/pixel/code/FrontierSplit/docs/context/DESIGN_CLOUD_ARCHITECTURE.md#23-the-decode-worker-small-gpu--edge) | [`frontiersplit/context/decode_worker.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/context/decode_worker.py) | [`tests/test_decode_worker.py`](file:///Users/pixel/code/FrontierSplit/tests/test_decode_worker.py) |
| **Context** | Delta Prefill & HF Patcher | [`docs/context/DESIGN_CLOUD_ARCHITECTURE.md`](file:///Users/pixel/code/FrontierSplit/docs/context/DESIGN_CLOUD_ARCHITECTURE.md#52-disaggregated-delta-prefill-algorithm) | [`frontiersplit/context/hf_model.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/context/hf_model.py) | [`tests/test_hf_model.py`](file:///Users/pixel/code/FrontierSplit/tests/test_hf_model.py)<br>[`tests/test_delta_prefill_and_ttl.py`](file:///Users/pixel/code/FrontierSplit/tests/test_delta_prefill_and_ttl.py) |
| **Context** | Radix Prefix Cache | [`docs/context/DESIGN_CLOUD_ARCHITECTURE.md`](file:///Users/pixel/code/FrontierSplit/docs/context/DESIGN_CLOUD_ARCHITECTURE.md#31-radix-tree-prefix-caching) | [`frontiersplit/context/prefix_cache.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/context/prefix_cache.py) | [`tests/test_prefix_cache.py`](file:///Users/pixel/code/FrontierSplit/tests/test_prefix_cache.py) |
| **Context** | Cluster Routing & Hashing | [`docs/context/DESIGN_CLOUD_ARCHITECTURE.md`](file:///Users/pixel/code/FrontierSplit/docs/context/DESIGN_CLOUD_ARCHITECTURE.md#32-distributed-context-fabric--sharding) | [`frontiersplit/context/routing.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/context/routing.py) | [`tests/test_routing.py`](file:///Users/pixel/code/FrontierSplit/tests/test_routing.py) |
| **Pipeline** | Model Weight Slicer | [`docs/pipeline/DESIGN_MIXTRAL_8X7B.md`](file:///Users/pixel/code/FrontierSplit/docs/pipeline/DESIGN_MIXTRAL_8X7B.md#2-hardware-allocation--memory-budget) | [`frontiersplit/pipeline/slicer.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/pipeline/slicer.py) | [`tests/test_slicer.py`](file:///Users/pixel/code/FrontierSplit/tests/test_slicer.py) |
| **Pipeline** | Shard Planner | [`docs/pipeline/DESIGN_MIXTRAL_8X7B.md`](file:///Users/pixel/code/FrontierSplit/docs/pipeline/DESIGN_MIXTRAL_8X7B.md#2-hardware-allocation--memory-budget) | [`frontiersplit/pipeline/shard_planner.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/pipeline/shard_planner.py) | [`tests/test_shard_planner.py`](file:///Users/pixel/code/FrontierSplit/tests/test_shard_planner.py) |
| **Pipeline** | Stage Forward Worker | [`docs/pipeline/DESIGN_KV_CACHE.md`](file:///Users/pixel/code/FrontierSplit/docs/pipeline/DESIGN_KV_CACHE.md#61-worker-state-structure-frontiersplitpipelineworkerpy) | [`frontiersplit/pipeline/worker.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/pipeline/worker.py) | [`tests/test_kv_cache.py`](file:///Users/pixel/code/FrontierSplit/tests/test_kv_cache.py)<br>[`tests/test_static_cache.py`](file:///Users/pixel/code/FrontierSplit/tests/test_static_cache.py) |
| **Pipeline** | 1F1B Micro-batch Scheduler | [`docs/pipeline/DESIGN_MIXTRAL_8X7B.md`](file:///Users/pixel/code/FrontierSplit/docs/pipeline/DESIGN_MIXTRAL_8X7B.md#31-pipeline-bubble-neutralization) | [`frontiersplit/pipeline/scheduler.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/pipeline/scheduler.py) | [`tests/test_scheduler.py`](file:///Users/pixel/code/FrontierSplit/tests/test_scheduler.py)<br>[`tests/test_1f1b_pipeline.py`](file:///Users/pixel/code/FrontierSplit/tests/test_1f1b_pipeline.py) |
| **Pipeline** | Binary Framing Protocol | [`docs/pipeline/DESIGN_KV_CACHE.md`](file:///Users/pixel/code/FrontierSplit/docs/pipeline/DESIGN_KV_CACHE.md#62-scheduler-session-protocol-frontiersplitpipelineprotocolpy) | [`frontiersplit/pipeline/protocol.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/pipeline/protocol.py) | [`tests/test_transport.py`](file:///Users/pixel/code/FrontierSplit/tests/test_transport.py) |
| **Pipeline** | Persistent TCP Transport | [`docs/pipeline/DESIGN_MIXTRAL_8X7B.md`](file:///Users/pixel/code/FrontierSplit/docs/pipeline/DESIGN_MIXTRAL_8X7B.md#32-binary-activation-protocol-and-transport) | [`frontiersplit/pipeline/transport.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/pipeline/transport.py) | [`tests/test_binary_pipeline.py`](file:///Users/pixel/code/FrontierSplit/tests/test_binary_pipeline.py) |
| **Pipeline** | Cloud TPU v5e Runner | [`docs/pipeline/DESIGN_MIXTRAL_8X7B.md`](file:///Users/pixel/code/FrontierSplit/docs/pipeline/DESIGN_MIXTRAL_8X7B.md#1-executive-summary) | [`frontiersplit/pipeline/tpu_runner.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/pipeline/tpu_runner.py) | — |
| **Gateway** | OpenAI API Gateway | [`docs/context/DESIGN_CLOUD_ARCHITECTURE.md`](file:///Users/pixel/code/FrontierSplit/docs/context/DESIGN_CLOUD_ARCHITECTURE.md#4-end-to-end-request-lifecycle) | [`frontiersplit/gateway.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/gateway.py) | [`tests/test_gateway.py`](file:///Users/pixel/code/FrontierSplit/tests/test_gateway.py) |

---

## 3. Disaggregated Context Engine Architecture

### The Weightless Context Machine
Unlike traditional inference workers that load weights and run GEMMs for full feed-forward layers:
- The **Context Server** loads **zero model weights**.
- 100% of VRAM/DRAM is allocated strictly to KV cache storage.
- It computes only $S = Q K^T / \sqrt{d_k}$, local online softmax statistics $(m, \ell)$, and accumulator $A = \operatorname{softmax}(S) V$.
- Zero-copy prefix slicing allows requests to reference slices of cached prefixes ($0 \dots \text{context\_len}$) with zero tensor duplication.
- Sliding session TTL (`ttl_seconds`) automatically evicts idle caches while refreshing active conversational sessions.

### Mathematical Invariant: Online Softmax Attention
Let $q \in \mathbb{R}^{1 \times d}$ be the current decoding query vector. The context KV is split into prompt slice $(K_p, V_p)$ on the remote node and output slice $(K_o, V_o)$ on the local worker.

Partial statistics from Context Server:
$$m_p = \max_j (S_p)_j, \quad \ell_p = \sum_j e^{(S_p)_j - m_p}, \quad A_p = \sum_j e^{(S_p)_j - m_p} (V_p)_j$$

Partial statistics from Decode Worker:
$$m_o = \max_j (S_o)_j, \quad \ell_o = \sum_j e^{(S_o)_j - m_o}, \quad A_o = \sum_j e^{(S_o)_j - m_o} (V_o)_j$$

Exact Rescaling Merge:
$$m_{\text{global}} = \max(m_p, m_o)$$
$$\ell_{\text{global}} = \ell_p \cdot e^{m_p - m_{\text{global}}} + \ell_o \cdot e^{m_o - m_{\text{global}}}$$
$$A_{\text{global}} = A_p \cdot e^{m_p - m_{\text{global}}} + A_o \cdot e^{m_o - m_{\text{global}}}$$
$$O = \frac{A_{\text{global}}}{\ell_{\text{global}}}$$

The merged output $O$ is **bit-for-bit mathematically identical** to computing full attention in a single monolithic pass.

---

## 4. Pipeline Parallel Engine Architecture

### Layer Slicing & Continuous 1F1B Scheduling
- For models that cannot fit in a single node's VRAM (e.g. Mixtral 8x7B FP16 needing ~94 GB VRAM across 4x L4 24GB GPUs):
  - Model layers are partitioned into contiguous blocks across nodes.
  - Intermediate hidden state activations ($[B, 1, 4096]$ FP16 $\approx 16\text{ KB}$ per token) are forwarded downstream over binary TCP.
  - The pipeline idle bubble is eliminated by interleaving multiple concurrent agent streams ($M \ge 4$) in a 1F1B (1-Forward-1-Backward equivalent) continuous cadence.
- For complete memory layouts and allocation algorithms, see [`docs/pipeline/DESIGN_KV_CACHE.md`](file:///Users/pixel/code/FrontierSplit/docs/pipeline/DESIGN_KV_CACHE.md).
