# FrontierSplit Documentation

This directory holds the architectural analysis and prescriptive engineering specifications for **FrontierSplit**.

## Current Documents

* **[`architecture.md`](architecture.md)** — Current primary system architecture for high-bandwidth `TPU` / `HBM` `GPU` execution (coarse `L_stage`-layer stages, 4-chip sharding, bucketed Online Softmax wave containers, piggybacked prefill, tiered Host-RAM KV caching, and Stage-1 correctness contracts).
* **[`evals.md`](evals.md)** — Evaluation suite (`LM Evaluation Harness`, `GuideLLM`; `RULER` deferred), reference parity methodology, and per-eval considerations.
* **[`API.md`](API.md)** — `vLLM`-compatible OpenAI HTTP and streaming interface specification.
* **[`paper/frontiersplit_technical_paper_draft2.md`](paper/frontiersplit_technical_paper_draft2.md)** — Analytical Roofline foundation, model/layer/KV byte anatomy, Online Softmax math, related-work comparison, and prior intra-layer `Attn+KV` $\to$ `FFN` disaggregation analysis for lower-bandwidth / local hardware.

## Project Conventions & Agent Rules

* **[`../AGENTS.md`](../AGENTS.md)** — Repository invariants, execution rules, testing standards, and code style.
* **[`../style_guide.md`](../style_guide.md)** — Standalone coding style reference (Python 3.14, PascalCase functions/methods, 4-group `fs_*` imports, keyword-only signatures, pure `torch.Tensor`).

*(Historical v0.1 prototype design notes are preserved on the `archive/v0.1-prototypes` Git branch).*
