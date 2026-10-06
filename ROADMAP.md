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

## Phase 3: Pipeline Transport & Ingress Gateway
- [ ] Build point-to-point tensor transport worker using gRPC / PyTorch distributed RPC.
- [ ] Implement layer-forwarding loop:
  - Node 0 (Input + Embeddings + Layers 0..N) -> Node 1 -> Node 2 -> Node 3 (Output Head).
- [ ] Implement an OpenAI-compatible API Gateway exposing `/v1/chat/completions`.
- [ ] Verify basic single-stream and multi-stream text generation on the 4-node cluster.

---

## Phase 4: Multi-Agent Concurrency & Bubble Elimination
- [ ] Implement concurrent request scheduler in the Ingress Gateway.
- [ ] Connect a multi-agent harness (DSPy / OpenHands / multi-agent reasoning trees).
- [ ] Build real-time telemetry dashboard (Prometheus / Grafana or lightweight CLI logger) tracking:
  - Per-node GPU utilization & memory bandwidth.
  - Network handoff latency (VPC ping & tensor serialization time).
  - Pipeline bubble factor under varying concurrent request counts ($N=1, 4, 8, 16$).

---

## Phase 5: Showcases & Evaluations
- [ ] **Living Demo**: Benchmark IFEval and SWE-bench Lite on the 4x L4 cluster.
- [ ] **Tier 2 (Google TPU Showcase)**: Port pipeline orchestration to Cloud TPU v5e (comparing ICI vs. VPC mesh).
- [ ] **Tier 3 (Flagship Benchmark Sprint)**: Execute scaled-out benchmark on a frontier 1.6T–2TB model.
