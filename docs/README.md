# FrontierSplit: Design & Architecture Portal

Welcome to the **FrontierSplit Design and Architecture Documentation**.

FrontierSplit decouples LLM inference across commodity hardware clusters using two complementary, independently reviewable techniques:
1. **Disaggregated Context Engine (Milestone 2 - Current Focus)**: Offloads prompt Key-Value caches to dedicated "weightless" memory nodes, computing exact partial attention remotely via **Online Softmax rescaling** with tiny 8 KB network payloads.
2. **1F1B Pipeline Parallel Engine (Milestone 1 - Historical)**: Slices large Mixture-of-Experts (MoE) models layer-by-layer across independent GPU/TPU nodes over VPC Ethernet.

---

## Design-to-Implementation Traceability Matrix

The table below maps every architectural specification directly to its production source code and test suite:

| Architectural Subsystem | Design Specification | Implementation Subpackage & Modules | Validation Test Suite |
| :--- | :--- | :--- | :--- |
| **Disaggregated Context Engine** | | [`frontiersplit/context/`](file:///Users/pixel/code/FrontierSplit/frontiersplit/context/) | |
| • Numerically Stable Online Softmax | [`docs/context/DESIGN_CLOUD_ARCHITECTURE.md`](file:///Users/pixel/code/FrontierSplit/docs/context/DESIGN_CLOUD_ARCHITECTURE.md#21-online-softmax-attention-math) (Sec 2.1) | [`online_softmax.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/context/online_softmax.py) (`compute_partial_attention`, `finalize_attention`, `merge_partial_attentions`) | [`test_online_softmax.py`](file:///Users/pixel/code/FrontierSplit/tests/test_online_softmax.py) |
| • Weightless Context Server & Sliding TTL | [`docs/context/DESIGN_CLOUD_ARCHITECTURE.md`](file:///Users/pixel/code/FrontierSplit/docs/context/DESIGN_CLOUD_ARCHITECTURE.md#22-disaggregated-prompt-serving--the-weightless-context-machine) (Sec 2.2, 5.1) | [`context_server.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/context/context_server.py) (`ContextServer`, `ContextClient`, `ContextStore`) | [`test_context_server.py`](file:///Users/pixel/code/FrontierSplit/tests/test_context_server.py)<br>[`test_delta_prefill_and_ttl.py`](file:///Users/pixel/code/FrontierSplit/tests/test_delta_prefill_and_ttl.py) |
| • Lightweight Decode Attention Worker | [`docs/context/DESIGN_CLOUD_ARCHITECTURE.md`](file:///Users/pixel/code/FrontierSplit/docs/context/DESIGN_CLOUD_ARCHITECTURE.md#23-the-decode-worker-small-gpu--edge) (Sec 2.3) | [`decode_worker.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/context/decode_worker.py) (`DecodeWorker`, `DisaggregatedAttention`, `LocalOutputKVCache`) | [`test_decode_worker.py`](file:///Users/pixel/code/FrontierSplit/tests/test_decode_worker.py) |
| • Disaggregated Delta Prefill & HF Patcher | [`docs/context/DESIGN_CLOUD_ARCHITECTURE.md`](file:///Users/pixel/code/FrontierSplit/docs/context/DESIGN_CLOUD_ARCHITECTURE.md#52-disaggregated-delta-prefill-algorithm) (Sec 5.2, 5.3) | [`hf_model.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/context/hf_model.py) (`DisaggregatedModel`, `DisaggregatedAttentionPatcher`) | [`test_hf_model.py`](file:///Users/pixel/code/FrontierSplit/tests/test_hf_model.py)<br>[`test_delta_prefill_and_ttl.py`](file:///Users/pixel/code/FrontierSplit/tests/test_delta_prefill_and_ttl.py) |
| • Radix Tree Prefix Caching | [`docs/context/DESIGN_CLOUD_ARCHITECTURE.md`](file:///Users/pixel/code/FrontierSplit/docs/context/DESIGN_CLOUD_ARCHITECTURE.md#31-radix-tree-prefix-caching) (Sec 3.1) | [`prefix_cache.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/context/prefix_cache.py) (`RadixPrefixCache`, `RadixNode`) | [`test_prefix_cache.py`](file:///Users/pixel/code/FrontierSplit/tests/test_prefix_cache.py) |
| • Consistent Hashing & Prefix-Affinity Routing | [`docs/context/DESIGN_CLOUD_ARCHITECTURE.md`](file:///Users/pixel/code/FrontierSplit/docs/context/DESIGN_CLOUD_ARCHITECTURE.md#32-distributed-context-fabric--sharding) (Sec 3.2, 4.1) | [`routing.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/context/routing.py) (`ConsistentHashRing`, `PrefixRouter`, `ContextShardNode`) | [`test_routing.py`](file:///Users/pixel/code/FrontierSplit/tests/test_routing.py) |
| **Pipeline Parallel Engine** | | [`frontiersplit/pipeline/`](file:///Users/pixel/code/FrontierSplit/frontiersplit/pipeline/) | |
| • Model Weight Slicing & Memory Profiling | [`docs/pipeline/DESIGN_MIXTRAL_8X7B.md`](file:///Users/pixel/code/FrontierSplit/docs/pipeline/DESIGN_MIXTRAL_8X7B.md#2-hardware-allocation--memory-budget) (Sec 2) | [`slicer.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/pipeline/slicer.py) (`ModelSlicer`)<br>[`shard_planner.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/pipeline/shard_planner.py) (`ShardPlanner`) | [`test_slicer.py`](file:///Users/pixel/code/FrontierSplit/tests/test_slicer.py)<br>[`test_shard_planner.py`](file:///Users/pixel/code/FrontierSplit/tests/test_shard_planner.py) |
| • 1F1B Pipeline Stage Worker & Static Cache | [`docs/pipeline/DESIGN_KV_CACHE.md`](file:///Users/pixel/code/FrontierSplit/docs/pipeline/DESIGN_KV_CACHE.md#61-worker-state-structure-frontiersplitpipelineworkerpy) (Sec 6.1) | [`worker.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/pipeline/worker.py) (`WorkerKVCacheStore`, `create_worker_app`) | [`test_kv_cache.py`](file:///Users/pixel/code/FrontierSplit/tests/test_kv_cache.py)<br>[`test_static_cache.py`](file:///Users/pixel/code/FrontierSplit/tests/test_static_cache.py) |
| • 1F1B Micro-batch Scheduler & Interleaving | [`docs/pipeline/DESIGN_MIXTRAL_8X7B.md`](file:///Users/pixel/code/FrontierSplit/docs/pipeline/DESIGN_MIXTRAL_8X7B.md#31-pipeline-bubble-neutralization) (Sec 3.1) | [`scheduler.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/pipeline/scheduler.py) (`PipelineScheduler`, `ScheduledRequest`) | [`test_scheduler.py`](file:///Users/pixel/code/FrontierSplit/tests/test_scheduler.py)<br>[`test_1f1b_pipeline.py`](file:///Users/pixel/code/FrontierSplit/tests/test_1f1b_pipeline.py) |
| • Persistent Binary TCP Transport & Framing | [`docs/pipeline/DESIGN_KV_CACHE.md`](file:///Users/pixel/code/FrontierSplit/docs/pipeline/DESIGN_KV_CACHE.md#62-scheduler-session-protocol-frontiersplitpipelineprotocolpy) (Sec 6.2) | [`protocol.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/pipeline/protocol.py) (`BatchedActivationPacket`)<br>[`transport.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/pipeline/transport.py) (`BinaryTransportClient`, `BinaryTransportServer`) | [`test_transport.py`](file:///Users/pixel/code/FrontierSplit/tests/test_transport.py)<br>[`test_binary_pipeline.py`](file:///Users/pixel/code/FrontierSplit/tests/test_binary_pipeline.py) |
| • Cloud TPU v5e Multi-Core Runner | [`docs/pipeline/DESIGN_MIXTRAL_8X7B.md`](file:///Users/pixel/code/FrontierSplit/docs/pipeline/DESIGN_MIXTRAL_8X7B.md#1-executive-summary) (Sec 1) | [`tpu_runner.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/pipeline/tpu_runner.py) (`xmp.spawn` stage runner) | — |
| **Common Ingress Gateway** | | [`frontiersplit/`](file:///Users/pixel/code/FrontierSplit/frontiersplit/) | |
| • OpenAI-Compatible REST API (/v1/chat/completions) | [`docs/context/DESIGN_CLOUD_ARCHITECTURE.md`](file:///Users/pixel/code/FrontierSplit/docs/context/DESIGN_CLOUD_ARCHITECTURE.md#4-end-to-end-request-lifecycle) (Sec 4) | [`gateway.py`](file:///Users/pixel/code/FrontierSplit/frontiersplit/gateway.py) (`create_gateway_app`, `ChatCompletionRequest`) | [`test_gateway.py`](file:///Users/pixel/code/FrontierSplit/tests/test_gateway.py)<br>[`test_pipeline.py`](file:///Users/pixel/code/FrontierSplit/tests/test_pipeline.py) |

---

## Document Index & Directory Layout

```text
docs/
├── README.md                              # This master index and traceability matrix
├── context/                               # Technique 2: Disaggregated Context Engine Design
│   ├── DESIGN_CLOUD_ARCHITECTURE.md       # Full architecture: Online Softmax, Delta Prefill, Sharding
│   └── PAPER_SYNTHESIS_DISAGGREGATED_ATTENTION.md # Foundations, hardware mapping & multi-tier chunking
└── pipeline/                              # Technique 1: Pipeline Parallel Engine Design
    ├── DESIGN_MIXTRAL_8X7B.md             # 8-stage Mixtral 8x7B MoE pipeline slicing specification
    └── DESIGN_KV_CACHE.md                 # Tile-aligned bounded KV cache allocation for pipeline workers
```
