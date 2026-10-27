# FrontierSplit Documentation

This directory holds the architectural analysis and prescriptive engineering specifications for **FrontierSplit**.

## Current Documents

* **[`paper/frontiersplit_technical_paper_draft2.md`](paper/frontiersplit_technical_paper_draft2.md)** — Analytical foundation, model/layer/KV byte anatomy, intra-layer `Attn+KV` $\to$ `FFN` partitioning, sequence chunking & interruption-interval striping, primary-source hardware Roofline modeling, and related-work comparison (`FlashDecoding`, `DistServe`, `Mooncake`, `vLLM afd-plugin` / `StepMesh`, `SGLang AFD`, `DeepInfer`).

## Project Conventions & Agent Rules

* **[`../AGENTS.md`](../AGENTS.md)** — Repository invariants, execution rules, testing standards, and code style.
* **[`../style_guide.md`](../style_guide.md)** — Standalone coding style reference (Python 3.14, PascalCase functions/methods, 4-group `fs_*` imports, keyword-only signatures, pure `torch.Tensor`).

*(Historical v0.1 prototype design notes are preserved on the `archive/v0.1-prototypes` Git branch).*
