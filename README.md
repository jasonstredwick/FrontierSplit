# FrontierSplit

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)

**FrontierSplit** is a modular Python/PyTorch library for **distributed unquantized 16-bit (`BF16`) LLM inference** on high-bandwidth **`TPU`** and **`GPU`** clusters, targeting **`Mistral-7B`** (Dense) and **`Mixtral-8x7B`** (MoE) as primary baseline models.

---

## Core Design Principles

1. **Coarse Multi-Layer Flow Stages & Intra-Host Sharding:** Partition models into contiguous full-layer stages sized above host/network submission overhead, sharded intra-host across `TPU` (`ICI`) or `GPU` (`NVLink`) chips and connected across stages via persistent zero-pickle binary TCP.
2. **Static-Shape KV Containers & `FP32` Online Softmax:** Manage session history in contiguous fixed-size KV blocks and execute mixed-bucket waves inside pre-compiled static containers via `FP32` Online Softmax reduction without runtime XLA graph recompilation.
3. **Piggybacked Chunked Prefill & Tiered KV Caching:** Pack bounded prefill chunks into unused systolic batch rows alongside live decode waves, and offload/reload idle multi-turn KV blocks between device `HBM` and pinned Host CPU RAM at block granularity.
4. **Explicit Roofline & Capacity Accounting:** Separate global pipeline concurrency ($M$) from stage batch size ($B$), resident weights from per-step active bytes read, and usable memory from physical stack sizes across high-bandwidth flow pipelines and prior low-bandwidth disaggregated topologies.

---

## Documentation & Conventions

* **[`docs/README.md`](docs/README.md)** — Documentation index.
* **[`docs/architecture.md`](docs/architecture.md)** — Primary system architecture for high-bandwidth `TPU` / `HBM` `GPU` flow execution and Stage-1 correctness contracts.
* **[`evals/README.md`](evals/README.md)** — Evaluation runner setup, configuration, and commands.
* **[`frontiersplit/models/experiments/README.md`](frontiersplit/models/experiments/README.md)** — Hugging Face reference and local baseline model experiments.
* **[`docs/paper/frontiersplit_technical_paper_draft2.md`](docs/paper/frontiersplit_technical_paper_draft2.md)** — Analytical Roofline foundations, Online Softmax math, and prior intra-layer `Attn+KV` $\to$ `FFN` disaggregation modeling for lower-bandwidth hardware.
* **[`AGENTS.md`](AGENTS.md)** & **[`style_guide.md`](style_guide.md)** — Workspace execution rules, architectural invariants, and Python 3.14 coding conventions.

---

## Development Setup & Running Commands

Requires **Python 3.14** in `.venv/`. Run scripts and tests from the repository root with bytecode routed to `.__pycache__/`:

The full test suite includes harness integration tests and requires the dependencies in
[`evals/requirements.lock`](evals/requirements.lock); installation is described in the [runner guide](evals/README.md).

```sh
# Direct script execution
PYTHONPATH=. .venv/bin/python -X pycache_prefix=.__pycache__ path/to/script.py

# Run test suite
PYTHONPATH=. .venv/bin/python -X pycache_prefix=.__pycache__ -m pytest
```

*(Historical v0.1 prototype modules and experiments are preserved on the `archive/v0.1-prototypes` Git branch.)*

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).

Portions of FrontierSplit are adapted from third-party open-source projects; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution and source references.
