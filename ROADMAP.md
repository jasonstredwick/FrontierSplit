# FrontierSplit: Project Roadmap

## Phase 1: Local Foundation & Model Slicer (Complete)
- [x] Configure Git repository, SSH credentials, and remote tracking.
- [x] Provision GCP Project (`frontiersplit-proto`), link billing, and configure spend alerts.
- [x] Verify GCP GPU quotas in `us-central1` (16x NVIDIA L4 confirmed).
- [x] Initialize Python virtual environment (`.venv`) and repository `.gitignore`.
- [x] Formalize `ARCHITECTURE.md` and `ROADMAP.md`.
- [x] Build **Model Slicer & Memory Profiler** (`frontiersplit/slicer.py`):
  - Read arbitrary HuggingFace model architectures (Mixtral 8x7B, Qwen2-57B-A14B, DeepSeek V2/V3).
  - Calculate exact parameter byte sizes for Attention ($Q, K, V, O$) and FFN/MoE matrices ($W_{gate}, W_{up}, W_{down}$).
  - Output partition plans for clusters of arbitrary GPU counts and VRAM capacities.

---

## Phase 2: Cloud Cluster Provisioning Automation (Complete)
- [x] Create reproducible cluster automation (`scripts/cluster_up.sh`):
  - 4x `g2-standard-4` instances with NVIDIA L4 (24GB VRAM each) in `us-central1-a`.
  - VPC network configuration with internal low-latency firewall rules (`frontiersplit-internal-mesh`).
  - Startup scripts for NVIDIA drivers, CUDA, PyTorch, and environment dependencies (`scripts/startup_node.sh`).
- [x] Implement start/stop/down automation to preserve credits when not testing (`scripts/cluster_stop.sh`, `scripts/cluster_down.sh`, `scripts/cluster_status.sh`).

---

## Phase 3: Pipeline Transport & Ingress Gateway (Complete)
- [x] Build point-to-point tensor transport protocol (`frontiersplit/protocol.py`).
- [x] Implement layer-forwarding pipeline worker (`frontiersplit/worker.py`):
  - Node 0 (Input + Embeddings + Layers 0..N) -> Node 1 -> Node 2 -> Node 3 (Output Head).
- [x] Implement an OpenAI-compatible API Gateway exposing `/v1/chat/completions` (`frontiersplit/gateway.py`).
- [x] Verify basic single-stream and multi-stream text generation through full 4-stage pipeline tests (`tests/test_pipeline.py`).

---

## Phase 4: Multi-Agent Concurrency & Bubble Elimination (Complete)
- [x] Implement concurrent request scheduler in the Ingress Gateway (`frontiersplit/scheduler.py`, `frontiersplit/gateway.py`):
  - Continuous micro-batch interleaving across pipeline stages.
  - Server-Sent Events (SSE) streaming support (`stream: true`).
- [x] Connect a multi-agent harness (`benchmarks/agent_harness.py`):
  - Tree-of-Thoughts (ToT) parallel reasoning branch exploration.
  - Autonomous agent swarm task dispatcher.
  - Concurrency sweep benchmark ($N=1, 4, 8, 16$).
- [x] Build real-time telemetry dashboard & bubble tracker (`benchmarks/telemetry_dashboard.py`):
  - Track active streams ($M$), aggregate throughput (tok/s), and per-stage latency.
  - Track empirical & theoretical pipeline bubble factors ($F_{bubble}$) under varying concurrency.
- [x] Comprehensive test suites (`tests/test_scheduler.py`, `tests/test_harness.py`).

---

## Phase 5: Showcases & Evaluations
- [x] **IFEval Benchmark Runner & Verifier** (`benchmarks/eval_ifeval.py`, `tests/test_eval_ifeval.py`):
  - Verifiable instruction constraints (word counts, JSON schemas, letter exclusions, casing, bullet points).
  - Strict and loose accuracy evaluation across concurrent multi-stream traffic.
- [x] **SWE-bench Lite Compound Agent Harness** (`benchmarks/eval_swebench.py`, `tests/test_eval_swebench.py`):
  - Autonomous multi-turn software agent loop (Issue Analysis -> Plan -> Unified Diff Generation -> Patch Verification).
  - Diff syntax validation and target file localization tracking under concurrent multi-agent dispatch.
- [x] **Unified Evaluation CLI & Markdown Reporter** (`benchmarks/run_evals.py`, `tests/test_run_evals.py`):
  - Correlates cognitive scores with real-time cluster telemetry (tok/s, latency, $F_{bubble}$ idle bubble reduction).
  - Automated report generation (`eval_results/eval_report.md`, `eval_results/eval_summary.json`).
- [x] **Live Cluster Validation & Saturation Sweep**: Executed full saturation curve ($M=1..32$) and 3-trial IFEval/SWE-bench evaluation on 8-node Mixtral 8x7B FP16 cluster (`experiments/20261008_mixtral8x7b_fp16_8x_t4/`).
- [ ] **Tier 2 (Google TPU Showcase)**: Port pipeline orchestration to Cloud TPU v5e (comparing ICI vs. VPC mesh).
- [ ] **Tier 3 (Flagship Benchmark Sprint)**: Execute scaled-out benchmark on a frontier 1.6T–2TB model.

---

## Phase 6: High-Throughput & Low-Latency Optimization Roadmap
Techniques identified for future re-evaluation to scale generation speeds and minimize step latency:
- [ ] **Binary Transport Layer Modernization (gRPC / Async Raw TCP)**:
  - Replace HTTP/JSON/base64 payload serialization with gRPC protobuf or custom async binary TCP sockets with memory-pinned PyTorch tensors.
  - Eliminates ~10-15 ms serialization overhead per hop, targeting < 1.5 ms hop transport and halving round-trip step latency from ~200 ms to ~100 ms.
- [ ] **Pipeline-Parallel Speculative Decoding**:
  - Deploy a lightweight draft model (e.g. 1B parameter dense model) on Stage 0 to propose $K=3-5$ candidate tokens.
  - Pipe candidate verification sequences through the pipeline in a single forward pass, accepting multiple verified tokens per round-trip.
  - Expected speedup: 2.5x–3.5x boost in single-stream generation speed without altering model weights.
- [ ] **Wire-Only Activation Quantization (Dynamic FP8 E4M3/E5M2)**:
  - Dynamically quantize intermediate hidden state tensors from FP16 to FP8 solely across the network wire, immediately dequantizing upon receipt on the next stage.
  - Halves inter-node network bandwidth (from 8 KB to 4 KB per token) while keeping all model weights strictly unquantized in FP16.
- [ ] **CUDA Graph Capture for Fixed Micro-Batches**:
  - Pre-capture static forward-pass execution graphs for discrete micro-batch sizes ($B \in [1, 2, 4, 8]$) on each stage.
  - Completely eliminates PyTorch/Python dispatch overhead and CPU-GPU synchronization bubbles inside each node's local 4-layer execution.
- [ ] **Asynchronous Prefill & Decode Overlapping**:
  - Decouple compute-bound prompt prefill bursts from latency-bound autoregressive decode steps so prefill micro-batches fill pipeline bubbles without starving active streams.
