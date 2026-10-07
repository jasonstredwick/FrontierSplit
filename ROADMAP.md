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
- [ ] **Living Demo Cluster Run**: Execute live benchmark sprint on the 4x L4 GCP cluster.
- [ ] **Tier 2 (Google TPU Showcase)**: Port pipeline orchestration to Cloud TPU v5e (comparing ICI vs. VPC mesh).
- [ ] **Tier 3 (Flagship Benchmark Sprint)**: Execute scaled-out benchmark on a frontier 1.6T–2TB model.
