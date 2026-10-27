# FrontierSplit

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)

**FrontierSplit** is a modular Python/PyTorch library and fleet-sizing toolkit for **distributed unquantized 16-bit (`BF16`/`FP16`) LLM inference** across commodity, older, and heterogeneous hardware, targeting **`Mistral-7B`** (Dense) and **`Mixtral-8x7B`** (MoE) as primary baseline models.

---

## Core Design Principles

1. **Compute-to-Data KV Disaggregation via Online Softmax:** Keep private per-session Key-Value caches pinned in memory while exchanging compact `8–32 KiB` single-token vectors over persistent zero-pickle binary TCP, merging sequence chunks via FP32 Online Softmax rescaling.
2. **Composable Layer & Intra-Layer Topologies:** Support consolidated execution, multi-layer pipeline slicing across small-VRAM GPUs, and intra-layer **`Attn+KV` $\to$ KV-Stateless `FFN`** disaggregation (`W_qkv` + aligned KV chunks + ragged-edge tail local to `Attn+KV`; coalesced batching over $\Delta t_{\text{wait}}$ + 1-token fan-out on `FFN`).
3. **Sequence Chunking & Interruption-Interval Striping:** Align historical KV blocks to fixed chunk boundaries, manage newly generated tokens in an active **ragged-edge** tail chunk, and expose a tunable **interruption-interval** chunk quantum supporting intra-request **self-striping** and inter-request **concurrency-striping**.
4. **Roofline-Guided Fleet Planning:** Model resident vs. active bytes, physical vs. software-usable VRAM, global in-flight concurrency ($M$) vs. stage batch size ($B$), and idealized weight-only crossover ($B_{\text{crit}} \approx F_{\text{peak}} / \text{BW}_{\text{peak}}$) to identify when to consolidate, when to pipeline, and when to disaggregate.

---

## Documentation & Conventions

* **[`docs/README.md`](docs/README.md)** — Documentation index.
* **[`docs/paper/frontiersplit_technical_paper_draft2.md`](docs/paper/frontiersplit_technical_paper_draft2.md)** — Full architectural analysis, intra-layer dataflow, chunking/striping mechanics, and primary-source hardware Roofline models.
* **[`AGENTS.md`](AGENTS.md)** & **[`style_guide.md`](style_guide.md)** — Workspace execution rules, architectural invariants, and Python 3.14 coding conventions.

---

## Development Setup & Running Commands

Requires **Python 3.14** in `.venv/`. Run scripts and tests from the repository root with bytecode routed to `.__pycache__/`:

```sh
# Direct script execution
PYTHONPATH=. .venv/bin/python -X pycache_prefix=.__pycache__ path/to/script.py

# Run test suite
PYTHONPATH=. .venv/bin/python -X pycache_prefix=.__pycache__ -m pytest
```

*(Note: `frontiersplit/` currently holds the v0.1 prototype modules backed up on branch `archive/v0.1-prototypes` while v2 prescriptive specifications and module refactoring are underway).*

---

## License

Apache License 2.0.
