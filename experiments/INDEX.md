# FrontierSplit: Experiment Registry

This registry tracks all cluster runs, evaluations, and hardware scaling experiments.

| Run ID | Date (UTC) | Model | Precision | Cluster Topology | Total VRAM | Est. Cost / hr | IFEval Strict | SWE-bench Diff Rate | Avg Tok/s | Saturation ($1 - F_{\text{bubble}}$) | Report |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `20261007_mistral7b_fp16_2x_t4` | 2026-10-07 | `mistralai/Mistral-7B-Instruct-v0.3` | FP16 | 2x NVIDIA T4 (`n1-standard-4`) | 32 GB | ~$0.70/hr | *Running* | *Running* | *Running* | *Running* | [Report](20261007_mistral7b_fp16_2x_t4/report/eval_report.md) |
