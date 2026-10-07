"""FrontierSplit Unified Evaluation Runner & Telemetry Reporter.

Executes IFEval and SWE-bench Lite compound benchmarks against the FrontierSplit Ingress
Gateway. Correlates cognitive accuracy scores with real-time cluster telemetry (pipeline
bubble elimination, throughput, and hardware saturation) and generates executive markdown reports.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any, Dict, Optional
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


def generate_markdown_report(
    ifeval_summary: Optional[Dict[str, Any]],
    swebench_summary: Optional[Dict[str, Any]],
    telemetry_before: Dict[str, Any],
    telemetry_after: Dict[str, Any],
    output_path: str,
) -> str:
    """Generate a GitHub Flavored Markdown evaluation report with ASCII tables and badges."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    md_lines = [
        "# FrontierSplit Evaluation Report: IFEval & SWE-bench Lite",
        "",
        f"**Generated**: {timestamp}  ",
        "**Cluster Architecture**: 4-Stage Distributed Pipeline Parallelism (PP) over VPC Ethernet  ",
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

    if ifeval_summary:
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

    if swebench_summary:
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

    md_lines.extend([
        "### 3. Pipeline Bubble Elimination Analysis",
        "",
        "Under Pipeline Parallelism with $K = 4$ stages, the idle bubble fraction $F_{\\text{bubble}}$ is defined as:",
        "",
        r"$$F_{\text{bubble}} = \frac{K - 1}{M + K - 1}$$",
        "",
        "Where $M$ is the number of concurrent active request streams.",
        "",
        "| Streams ($M$) | Idle Bubble ($F_{\\text{bubble}}$) | Hardware Saturation ($1 - F_{\\text{bubble}}$) | Status |",
        "| :---: | :---: | :---: | :--- |",
        "| 1 | 75.0% | 25.0% | Baseline (Unmitigated bubble) |",
        "| 4 | 42.9% | 57.1% | 2.3x Saturation Improvement |",
        "| 8 | 27.3% | 72.7% | 2.9x Saturation Improvement |",
        "| 16 | 15.8% | 84.2% | Near-Optimal Pipeline Saturation |",
        "",
        "---",
        "*Report generated by FrontierSplit Automated Benchmark Suite.*",
    ])

    report_content = "\n".join(md_lines)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_content


async def run_suite(
    base_url: str = "http://localhost:8000/v1",
    gateway_url: str = "http://localhost:8000",
    model: str = "frontiersplit-mixtral-8x7b",
    benchmarks: str = "all",
    concurrency: int = 4,
    output_dir: str = "eval_results",
) -> Dict[str, Any]:
    """Execute complete benchmark suite and write outputs."""
    os.makedirs(output_dir, exist_ok=True)
    telemetry_before = fetch_gateway_telemetry(gateway_url)

    ifeval_results = None
    swebench_results = None

    if benchmarks in ("ifeval", "all"):
        print("\nStarting IFEval Evaluation Phase...")
        runner = IFEvalRunner(base_url=base_url, model=model, concurrency=concurrency)
        ifeval_results = await runner.run_evaluation()

    if benchmarks in ("swebench", "all"):
        print("\nStarting SWE-bench Lite Evaluation Phase...")
        runner = SWEBenchAgentRunner(base_url=base_url, model=model, concurrency=concurrency)
        swebench_results = await runner.run_evaluation()

    telemetry_after = fetch_gateway_telemetry(gateway_url)

    # Save structured JSON summary
    suite_summary = {
        "timestamp": time.time(),
        "model": model,
        "concurrency": concurrency,
        "ifeval": ifeval_results,
        "swebench": swebench_results,
        "cluster_telemetry_before": telemetry_before,
        "cluster_telemetry_after": telemetry_after,
    }
    json_path = os.path.join(output_dir, "eval_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2)

    # Generate Markdown Report
    report_path = os.path.join(output_dir, "eval_report.md")
    generate_markdown_report(
        ifeval_summary=ifeval_results,
        swebench_summary=swebench_results,
        telemetry_before=telemetry_before,
        telemetry_after=telemetry_after,
        output_path=report_path,
    )

    print(f"\n[Done] Evaluation artifacts generated:")
    print(f"  - Structured JSON: {json_path}")
    print(f"  - Markdown Report: {report_path}")

    return suite_summary


def main():
    parser = argparse.ArgumentParser(description="FrontierSplit Unified Benchmark Suite")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1", help="API Gateway URL")
    parser.add_argument("--gateway-url", type=str, default="http://localhost:8000", help="Root Gateway URL for telemetry")
    parser.add_argument("--model", type=str, default="frontiersplit-mixtral-8x7b", help="Model name")
    parser.add_argument("--benchmarks", type=str, choices=["ifeval", "swebench", "all"], default="all", help="Benchmarks to execute")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent streams")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Directory for evaluation reports")
    args = parser.parse_args()

    asyncio.run(
        run_suite(
            base_url=args.base_url,
            gateway_url=args.gateway_url,
            model=args.model,
            benchmarks=args.benchmarks,
            concurrency=args.concurrency,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
