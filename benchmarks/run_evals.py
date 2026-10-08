"""FrontierSplit Unified Evaluation Runner & Statistical Telemetry Reporter.

Executes IFEval and SWE-bench Lite compound benchmarks against the FrontierSplit Ingress
Gateway across N trials for statistical relevance (mean, standard deviation, min, max, median).
Correlates cognitive accuracy scores with real-time cluster telemetry (pipeline bubble
elimination, throughput, and hardware saturation) and generates executive markdown reports.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from typing import Any, Dict, List, Optional
import requests

from benchmarks.eval_ifeval import IFEvalRunner
from benchmarks.eval_swebench import SWEBenchAgentRunner


def fetch_gateway_telemetry(gateway_url: str) -> Dict[str, Any]:
    """Fetch current pipeline telemetry from the Ingress Gateway."""
    base = gateway_url.rstrip("/")
    try:
        resp = requests.get(f"{base}/v1/telemetry", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def compute_stats(vals: List[float]) -> Dict[str, float]:
    """Compute summary statistics (mean, std dev, min, max, median) for a list of values."""
    if not vals:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    mean = sum(vals) / len(vals)
    variance = sum((x - mean) ** 2 for x in vals) / len(vals) if len(vals) > 1 else 0.0
    std = math.sqrt(variance)
    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
        "median": round(float(statistics.median(vals)), 2),
    }


def generate_markdown_report(
    ifeval_summary: Optional[Dict[str, Any]] = None,
    swebench_summary: Optional[Dict[str, Any]] = None,
    runs_data: Optional[List[Dict[str, Any]]] = None,
    aggregated_stats: Optional[Dict[str, Any]] = None,
    telemetry_before: Optional[Dict[str, Any]] = None,
    telemetry_after: Optional[Dict[str, Any]] = None,
    total_stages: int = 2,
    num_runs: int = 1,
    output_path: str = "eval_results/eval_report.md",
) -> str:
    """Generate a GitHub Flavored Markdown evaluation report with statistical tables and badges."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    telemetry_before = telemetry_before or {}
    telemetry_after = telemetry_after or {}

    k = total_stages
    md_lines = [
        f"# FrontierSplit Evaluation Report: IFEval & SWE-bench Lite ({num_runs} Runs)",
        "",
        f"**Generated**: {timestamp}  ",
        f"**Cluster Architecture**: {k}-Stage Distributed Pipeline Parallelism (PP) over VPC Ethernet  ",
        f"**Statistical Trials ($N$)**: `{num_runs}` repeated iterations for statistical relevance  ",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "FrontierSplit was evaluated on compound agentic workflows and strict instruction following. "
        "Unlike traditional single-stream benchmarks that incur high pipeline bubbles, concurrent "
        "multi-agent evaluation keeps the distributed GPU pipeline saturated.",
        "",
    ]

    # IFEval Section
    if aggregated_stats and "ifeval" in aggregated_stats:
        stats = aggregated_stats["ifeval"]
        md_lines.extend([
            "### 1. IFEval Statistical Results (Instruction Following)",
            "",
            f"Evaluated across **{num_runs} independent trials** with verifiable constraints:",
            "",
            "| Metric | Mean ± Std Dev | Min | Max | Median | Baseline / Reference |",
            "| :--- | :---: | :---: | :---: | :---: | :--- |",
            f"| **Strict Prompt Accuracy** | `{stats['strict_prompt_accuracy']['mean']}% ± {stats['strict_prompt_accuracy']['std']}%` | `{stats['strict_prompt_accuracy']['min']}%` | `{stats['strict_prompt_accuracy']['max']}%` | `{stats['strict_prompt_accuracy']['median']}%` | Strict raw prompt adherence |",
            f"| **Loose Prompt Accuracy** | `{stats['loose_prompt_accuracy']['mean']}% ± {stats['loose_prompt_accuracy']['std']}%` | `{stats['loose_prompt_accuracy']['min']}%` | `{stats['loose_prompt_accuracy']['max']}%` | `{stats['loose_prompt_accuracy']['median']}%` | Whitespace/markdown-tolerant |",
            f"| **Strict Instruction Accuracy** | `{stats['strict_instruction_accuracy']['mean']}% ± {stats['strict_instruction_accuracy']['std']}%` | `{stats['strict_instruction_accuracy']['min']}%` | `{stats['strict_instruction_accuracy']['max']}%` | `{stats['strict_instruction_accuracy']['median']}%` | Individual constraint verification |",
            f"| **Loose Instruction Accuracy** | `{stats['loose_instruction_accuracy']['mean']}% ± {stats['loose_instruction_accuracy']['std']}%` | `{stats['loose_instruction_accuracy']['min']}%` | `{stats['loose_instruction_accuracy']['max']}%` | `{stats['loose_instruction_accuracy']['median']}%` | Loose individual constraints |",
            f"| **Throughput (tok/s)** | `{stats['throughput_tok_per_sec']['mean']} ± {stats['throughput_tok_per_sec']['std']}` | `{stats['throughput_tok_per_sec']['min']}` | `{stats['throughput_tok_per_sec']['max']}` | `{stats['throughput_tok_per_sec']['median']}` | Concurrent throughput |",
            f"| **Pipeline Saturation** | `{stats['bubble_elimination_pct']['mean']}% ± {stats['bubble_elimination_pct']['std']}%` | `{stats['bubble_elimination_pct']['min']}%` | `{stats['bubble_elimination_pct']['max']}%` | `{stats['bubble_elimination_pct']['median']}%` | Hardware utilization |",
            "",
        ])
    elif ifeval_summary:
        md_lines.extend([
            "### 1. IFEval (Instruction-Following Evaluation)",
            "",
            "| Metric | Result | Target / Standard |",
            "| :--- | :--- | :--- |",
            f"| **Strict Prompt Accuracy** | `{ifeval_summary['strict_prompt_accuracy']}%` | Strict verification on raw output |",
            f"| **Loose Prompt Accuracy** | `{ifeval_summary['loose_prompt_accuracy']}%` | Whitespace/markdown-tolerant verification |",
            f"| **Strict Instruction Accuracy** | `{ifeval_summary['strict_instruction_accuracy']}%` | Individual constraint verification |",
            f"| **Loose Instruction Accuracy** | `{ifeval_summary['loose_instruction_accuracy']}%` | Loose individual constraints |",
            f"| **Total Prompts Evaluated** | `{ifeval_summary['total_prompts']}` | Curated representative suite |",
            f"| **Throughput** | `{ifeval_summary['throughput_tok_per_sec']} tok/s` | Concurrent throughput |",
            f"| **Pipeline Saturation** | `{ifeval_summary['bubble_elimination_pct']}%` | Hardware utilization |",
            "",
        ])

    # SWE-bench Section
    if aggregated_stats and "swebench" in aggregated_stats:
        s_stats = aggregated_stats["swebench"]
        md_lines.extend([
            "### 2. SWE-bench Lite Statistical Results (Autonomous Software Engineering)",
            "",
            f"Evaluated across **{num_runs} independent trials** with autonomous patch synthesis:",
            "",
            "| Metric | Mean ± Std Dev | Min | Max | Median | Notes |",
            "| :--- | :---: | :---: | :---: | :---: | :--- |",
            f"| **Valid Unified Diff Rate** | `{s_stats['valid_diff_rate_pct']['mean']}% ± {s_stats['valid_diff_rate_pct']['std']}%` | `{s_stats['valid_diff_rate_pct']['min']}%` | `{s_stats['valid_diff_rate_pct']['max']}%` | `{s_stats['valid_diff_rate_pct']['median']}%` | Syntax-valid git diffs |",
            f"| **Target File Hit Rate** | `{s_stats['target_files_matched_pct']['mean']}% ± {s_stats['target_files_matched_pct']['std']}%` | `{s_stats['target_files_matched_pct']['min']}%` | `{s_stats['target_files_matched_pct']['max']}%` | `{s_stats['target_files_matched_pct']['median']}%` | Correct buggy file localization |",
            f"| **Throughput (tok/s)** | `{s_stats['throughput_tok_per_sec']['mean']} ± {s_stats['throughput_tok_per_sec']['std']}` | `{s_stats['throughput_tok_per_sec']['min']}` | `{s_stats['throughput_tok_per_sec']['max']}` | `{s_stats['throughput_tok_per_sec']['median']}` | Multi-turn generation rate |",
            f"| **Pipeline Saturation** | `{s_stats['bubble_elimination_pct']['mean']}% ± {s_stats['bubble_elimination_pct']['std']}%` | `{s_stats['bubble_elimination_pct']['min']}%` | `{s_stats['bubble_elimination_pct']['max']}%` | `{s_stats['bubble_elimination_pct']['median']}%` | Hardware utilization |",
            "",
        ])
    elif swebench_summary:
        md_lines.extend([
            "### 2. SWE-bench Lite (Autonomous Software Engineering)",
            "",
            "| Metric | Result | Notes |",
            "| :--- | :--- | :--- |",
            f"| **Valid Unified Diff Rate** | `{swebench_summary['valid_diff_rate_pct']}%` ({swebench_summary['valid_diff_count']}/{swebench_summary['total_instances']}) | Proper hunk headers & diff syntax |",
            f"| **Target File Hit Rate** | `{swebench_summary['target_files_matched_pct']}%` ({swebench_summary['target_files_matched_count']}/{swebench_summary['total_instances']}) | Agent located correct buggy files |",
            f"| **Total Issues Evaluated** | `{swebench_summary['total_instances']}` | Multi-repo instances (Django, Requests, Flask) |",
            f"| **Throughput** | `{swebench_summary['throughput_tok_per_sec']} tok/s` | Sustained multi-turn generation |",
            f"| **Pipeline Saturation** | `{swebench_summary['bubble_elimination_pct']}%` | Hardware utilization |",
            "",
        ])

    # Trial Breakdown Table (if multiple runs)
    if runs_data and len(runs_data) > 1:
        md_lines.extend([
            "### 3. Per-Trial Iteration Breakdown",
            "",
            "| Trial | IFEval Strict % | IFEval Loose % | SWE-bench Diff % | Throughput (tok/s) | Saturation % | Duration (s) |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for r_idx, r in enumerate(runs_data, 1):
            if_r = r.get("ifeval") or {}
            sw_r = r.get("swebench") or {}
            strict_acc = f"{if_r.get('strict_prompt_accuracy', 'N/A')}%"
            loose_acc = f"{if_r.get('loose_prompt_accuracy', 'N/A')}%"
            diff_rate = f"{sw_r.get('valid_diff_rate_pct', 'N/A')}%"
            tp = if_r.get("throughput_tok_per_sec") or sw_r.get("throughput_tok_per_sec", 0.0)
            sat = if_r.get("bubble_elimination_pct") or sw_r.get("bubble_elimination_pct", 0.0)
            dur = round(r.get("duration_s", 0.0), 2)
            md_lines.append(f"| Run {r_idx} | {strict_acc} | {loose_acc} | {diff_rate} | {tp} | {sat}% | {dur}s |")
        md_lines.append("")

    # Pipeline Bubble Section
    md_lines.extend([
        f"### 4. Pipeline Bubble Elimination Analysis ($K = {k}$ Stages)",
        "",
        f"Under Pipeline Parallelism with $K = {k}$ stages, the idle bubble fraction $F_{{\\text{{bubble}}}}$ is defined as:",
        "",
        r"$$F_{\text{bubble}} = \frac{K - 1}{M + K - 1}$$",
        "",
        "Where $M$ is the number of concurrent active request streams.",
        "",
        "| Streams ($M$) | Idle Bubble ($F_{\\text{bubble}}$) | Hardware Saturation ($1 - F_{\\text{bubble}}$) | Status |",
        "| :---: | :---: | :---: | :--- |",
    ])

    for m in [1, 2, 4, 8, 16]:
        fb = (k - 1) / (m + k - 1)
        sat_pct = (1.0 - fb) * 100.0
        fb_pct = fb * 100.0
        if m == 1:
            status = "Baseline (Unmitigated bubble)"
        elif m <= 4:
            status = f"{round(sat_pct / ((1.0 - (k-1)/k)*100), 1)}x Saturation Improvement"
        else:
            status = "Near-Optimal Pipeline Saturation"
        md_lines.append(f"| {m} | {fb_pct:.1f}% | {sat_pct:.1f}% | {status} |")

    md_lines.extend([
        "",
        "---",
        f"*Report generated by FrontierSplit Automated Benchmark Suite ({num_runs} statistical trials).* ",
    ])

    report_content = "\n".join(md_lines)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_content


async def run_single_pass(
    base_url: str,
    gateway_url: str,
    model: str,
    benchmarks: str,
    concurrency: int,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """Execute a single pass of the selected benchmarks."""
    run_start = time.time()
    telemetry_before = fetch_gateway_telemetry(gateway_url)

    ifeval_results = None
    swebench_results = None

    if benchmarks in ("ifeval", "all"):
        print("\n  [Trial] Starting IFEval Evaluation Phase...")
        runner = IFEvalRunner(base_url=base_url, model=model, concurrency=concurrency)
        ifeval_results = await runner.run_evaluation(max_tokens=max_tokens)

    if benchmarks in ("swebench", "all"):
        print("\n  [Trial] Starting SWE-bench Lite Evaluation Phase...")
        runner = SWEBenchAgentRunner(base_url=base_url, model=model, concurrency=concurrency)
        tokens_per_step = max(128, max_tokens // 4)
        swebench_results = await runner.run_evaluation(tokens_per_step=tokens_per_step)


    telemetry_after = fetch_gateway_telemetry(gateway_url)
    duration_s = time.time() - run_start

    return {
        "timestamp": time.time(),
        "duration_s": duration_s,
        "ifeval": ifeval_results,
        "swebench": swebench_results,
        "cluster_telemetry_before": telemetry_before,
        "cluster_telemetry_after": telemetry_after,
    }


async def run_suite(
    base_url: str = "http://localhost:8000/v1",
    gateway_url: str = "http://localhost:8000",
    model: str = "frontiersplit-mixtral-8x7b",
    benchmarks: str = "all",
    concurrency: int = 4,
    num_runs: int = 1,
    max_tokens: int = 512,
    output_dir: str = "eval_results",
    experiment_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete benchmark suite across N runs and write statistical outputs."""
    os.makedirs(output_dir, exist_ok=True)
    runs_data: List[Dict[str, Any]] = []

    print(f"\n============================================================================")
    print(f" FrontierSplit Benchmark Suite: {num_runs} Runs (Concurrency M={concurrency})")
    print(f" Model: {model} | Benchmarks: {benchmarks} | Max Tokens: {max_tokens}")
    print(f"============================================================================")

    for run_idx in range(1, num_runs + 1):
        print(f"\n>>> Running Trial {run_idx}/{num_runs} ...")
        trial_data = await run_single_pass(
            base_url=base_url,
            gateway_url=gateway_url,
            model=model,
            benchmarks=benchmarks,
            concurrency=concurrency,
            max_tokens=max_tokens,
        )
        trial_data["run_index"] = run_idx
        runs_data.append(trial_data)

    # Determine total stages from gateway telemetry if available
    first_telemetry = runs_data[0].get("cluster_telemetry_after") or runs_data[0].get("cluster_telemetry_before") or {}
    total_stages = first_telemetry.get("total_stages", 2)

    # Compute statistical aggregation across runs
    aggregated_stats: Dict[str, Any] = {}

    if benchmarks in ("ifeval", "all"):
        if_keys = [
            "strict_prompt_accuracy",
            "loose_prompt_accuracy",
            "strict_instruction_accuracy",
            "loose_instruction_accuracy",
            "throughput_tok_per_sec",
            "bubble_elimination_pct",
        ]
        aggregated_stats["ifeval"] = {}
        for k in if_keys:
            vals = [float(r["ifeval"][k]) for r in runs_data if r.get("ifeval") and k in r["ifeval"]]
            aggregated_stats["ifeval"][k] = compute_stats(vals)

    if benchmarks in ("swebench", "all"):
        sw_keys = [
            "valid_diff_rate_pct",
            "target_files_matched_pct",
            "throughput_tok_per_sec",
            "bubble_elimination_pct",
        ]
        aggregated_stats["swebench"] = {}
        for k in sw_keys:
            vals = [float(r["swebench"][k]) for r in runs_data if r.get("swebench") and k in r["swebench"]]
            aggregated_stats["swebench"][k] = compute_stats(vals)

    # Compile master suite summary
    suite_summary = {
        "timestamp": time.time(),
        "model": model,
        "concurrency": concurrency,
        "num_runs": num_runs,
        "total_stages": total_stages,
        "aggregated_statistics": aggregated_stats,
        "runs": runs_data,
        "ifeval": runs_data[-1].get("ifeval"),
        "swebench": runs_data[-1].get("swebench"),
        "cluster_telemetry_before": runs_data[0].get("cluster_telemetry_before"),
        "cluster_telemetry_after": runs_data[-1].get("cluster_telemetry_after"),
    }

    json_path = os.path.join(output_dir, "eval_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2)

    report_path = os.path.join(output_dir, "eval_report.md")
    generate_markdown_report(
        ifeval_summary=runs_data[-1].get("ifeval"),
        swebench_summary=runs_data[-1].get("swebench"),
        runs_data=runs_data,
        aggregated_stats=aggregated_stats,
        telemetry_before=runs_data[0].get("cluster_telemetry_before"),
        telemetry_after=runs_data[-1].get("cluster_telemetry_after"),
        total_stages=total_stages,
        num_runs=num_runs,
        output_path=report_path,
    )

    # If an experiment directory is designated, mirror artifacts there
    if experiment_dir:
        report_exp_dir = os.path.join(experiment_dir, "report")
        bench_exp_dir = os.path.join(experiment_dir, "benchmarks")
        os.makedirs(report_exp_dir, exist_ok=True)
        os.makedirs(bench_exp_dir, exist_ok=True)

        with open(os.path.join(report_exp_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
            json.dump(suite_summary, f, indent=2)
        generate_markdown_report(
            ifeval_summary=runs_data[-1].get("ifeval"),
            swebench_summary=runs_data[-1].get("swebench"),
            runs_data=runs_data,
            aggregated_stats=aggregated_stats,
            telemetry_before=runs_data[0].get("cluster_telemetry_before"),
            telemetry_after=runs_data[-1].get("cluster_telemetry_after"),
            total_stages=total_stages,
            num_runs=num_runs,
            output_path=os.path.join(report_exp_dir, "eval_report.md"),
        )

        for r_idx, r in enumerate(runs_data, 1):
            if r.get("ifeval"):
                with open(os.path.join(bench_exp_dir, f"ifeval_traces_run_{r_idx}.json"), "w", encoding="utf-8") as f:
                    json.dump(r["ifeval"], f, indent=2)
            if r.get("swebench"):
                with open(os.path.join(bench_exp_dir, f"swebench_patches_run_{r_idx}.json"), "w", encoding="utf-8") as f:
                    json.dump(r["swebench"], f, indent=2)

    print(f"\n[Done] Evaluation artifacts generated for {num_runs} runs:")
    print(f"  - Structured JSON: {json_path}")
    print(f"  - Markdown Report: {report_path}")
    if experiment_dir:
        print(f"  - Archived in experiment directory: {experiment_dir}")

    return suite_summary


def main():
    parser = argparse.ArgumentParser(description="FrontierSplit Unified Benchmark Suite")
    parser.add_argument("--base-url", type=str, default=None, help="API Gateway URL")
    parser.add_argument("--gateway-url", type=str, default=None, help="Root Gateway URL for telemetry")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--benchmarks", type=str, choices=["ifeval", "swebench", "all"], default="all", help="Benchmarks to execute")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent streams")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of repeated runs for statistical relevance")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max output tokens for generations")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Directory for evaluation reports")
    parser.add_argument("--experiment-dir", type=str, default=None, help="Path to siloed experiment directory")
    args = parser.parse_args()

    # Auto-load from cluster_config.json if URLs not explicitly passed
    base_url = args.base_url
    gateway_url = args.gateway_url
    model = args.model

    cfg_path = os.environ.get("CLUSTER_CONFIG", "cluster_config.json")
    if not os.path.isabs(cfg_path):
        candidate_paths = [
            cfg_path,
            os.path.join(os.path.dirname(__file__), "..", cfg_path),
            os.path.join(os.getcwd(), cfg_path),
        ]
    else:
        candidate_paths = [cfg_path]

    for p in candidate_paths:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    gw = cfg.get("gateway", {})
                    host = gw.get("host", "127.0.0.1")
                    port = gw.get("port", 8000)
                    if base_url is None:
                        base_url = f"http://{host}:{port}/v1"
                    if gateway_url is None:
                        gateway_url = f"http://{host}:{port}"
                    if model is None:
                        model = cfg.get("model_id", "frontiersplit-mixtral-8x7b")
                    break
            except Exception:
                pass

    base_url = base_url or "http://localhost:8000/v1"
    gateway_url = gateway_url or "http://localhost:8000"
    model = model or "frontiersplit-mixtral-8x7b"

    # Default experiment directory based on model
    exp_dir = args.experiment_dir
    if not exp_dir:
        exp_id = "20261008_mixtral8x7b_fp16_8x_t4" if "mixtral" in model.lower() else "20261007_mistral7b_fp16_2x_t4"
        exp_dir = os.path.join(os.getcwd(), "experiments", exp_id)

    asyncio.run(
        run_suite(
            base_url=base_url,
            gateway_url=gateway_url,
            model=model,
            benchmarks=args.benchmarks,
            concurrency=args.concurrency,
            num_runs=args.num_runs,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
            experiment_dir=exp_dir,
        )
    )



if __name__ == "__main__":
    main()
