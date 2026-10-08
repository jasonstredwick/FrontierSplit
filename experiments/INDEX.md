# FrontierSplit: Experiment Registry

This registry tracks all cluster runs, evaluations, and hardware scaling experiments.

| Run ID | Date (UTC) | Model | Precision | Cluster Topology | Total VRAM | Est. Cost / hr | IFEval Strict | SWE-bench Diff Rate | Avg Tok/s | Saturation ($1 - F_{\text{bubble}}$) | Report |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `20261007_mistral7b_fp16_2x_t4` | 2026-10-07 | `mistralai/Mistral-7B-Instruct-v0.3` | FP16 | 2x NVIDIA T4 (`n1-standard-4`) | 32 GB | ~$0.70/hr | `40.0% ± 0.0%` (Inst: `53.85%`) | `100.0% ± 0.0%` | `13.74 (Peak Batched)` | `94.1% (at M=16)` | [Report](20261007_mistral7b_fp16_2x_t4/report/eval_report.md), [Saturation](20261007_mistral7b_fp16_2x_t4/report/saturation_curve.md) |
| `20261008_mixtral8x7b_fp16_8x_t4` | 2026-10-08 | `mistralai/Mixtral-8x7B-Instruct-v0.1` | FP16 | 8x NVIDIA T4 (`n1-standard-8`) | 128 GB | ~$2.80/hr | `60.0% ± 0.0%` (Inst: `69.23%`) | `100.0% ± 0.0%` | `31.09 (Peak Batched)` | `82.1% (at M=32)` | [Report](20261008_mixtral8x7b_fp16_8x_t4/report/eval_report.md), [Saturation](20261008_mixtral8x7b_fp16_8x_t4/report/saturation_curve.md) |



