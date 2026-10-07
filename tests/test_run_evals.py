"""Tests for FrontierSplit Unified Evaluation Runner & Markdown Reporter."""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from benchmarks.run_evals import generate_markdown_report, run_suite


class TestRunEvals(unittest.TestCase):
    def test_generate_markdown_report(self):
        ifeval_dummy = {
            "strict_prompt_accuracy": 90.0,
            "loose_prompt_accuracy": 95.0,
            "strict_instruction_accuracy": 92.5,
            "loose_instruction_accuracy": 96.0,
            "total_prompts": 10,
            "throughput_tok_per_sec": 450.0,
            "bubble_elimination_pct": 57.1,
        }
        swebench_dummy = {
            "valid_diff_rate_pct": 80.0,
            "valid_diff_count": 4,
            "total_instances": 5,
            "target_files_matched_pct": 80.0,
            "target_files_matched_count": 4,
            "throughput_tok_per_sec": 520.0,
            "bubble_elimination_pct": 57.1,
        }

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".md") as tf:
            report_path = tf.name

        try:
            content = generate_markdown_report(
                ifeval_summary=ifeval_dummy,
                swebench_summary=swebench_dummy,
                telemetry_before={},
                telemetry_after={},
                output_path=report_path,
            )

            self.assertIn("FrontierSplit Evaluation Report", content)
            self.assertIn("90.0%", content)
            self.assertIn("SWE-bench Lite", content)
            self.assertIn("80.0%", content)
            self.assertTrue(os.path.exists(report_path))
        finally:
            if os.path.exists(report_path):
                os.remove(report_path)

    def test_run_suite_mocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("benchmarks.eval_ifeval.IFEvalRunner.run_evaluation", new_callable=AsyncMock) as mock_ifeval, \
                 patch("benchmarks.eval_swebench.SWEBenchAgentRunner.run_evaluation", new_callable=AsyncMock) as mock_swebench:

                mock_ifeval.return_value = {
                    "strict_prompt_accuracy": 100.0,
                    "loose_prompt_accuracy": 100.0,
                    "strict_instruction_accuracy": 100.0,
                    "loose_instruction_accuracy": 100.0,
                    "total_prompts": 2,
                    "throughput_tok_per_sec": 200.0,
                    "bubble_elimination_pct": 57.1,
                }
                mock_swebench.return_value = {
                    "valid_diff_rate_pct": 100.0,
                    "valid_diff_count": 1,
                    "total_instances": 1,
                    "target_files_matched_pct": 100.0,
                    "target_files_matched_count": 1,
                    "throughput_tok_per_sec": 250.0,
                    "bubble_elimination_pct": 57.1,
                }

                summary = asyncio.run(
                    run_suite(
                        base_url="http://mock-gw/v1",
                        gateway_url="http://mock-gw",
                        benchmarks="all",
                        output_dir=tmpdir,
                    )
                )

                self.assertIsNotNone(summary["ifeval"])
                self.assertIsNotNone(summary["swebench"])
                self.assertTrue(os.path.exists(os.path.join(tmpdir, "eval_summary.json")))
                self.assertTrue(os.path.exists(os.path.join(tmpdir, "eval_report.md")))

    def test_run_suite_multi_run_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("benchmarks.eval_ifeval.IFEvalRunner.run_evaluation", new_callable=AsyncMock) as mock_ifeval:
                mock_ifeval.side_effect = [
                    {"strict_prompt_accuracy": 60.0, "loose_prompt_accuracy": 70.0, "strict_instruction_accuracy": 65.0, "loose_instruction_accuracy": 75.0, "throughput_tok_per_sec": 100.0, "bubble_elimination_pct": 50.0},
                    {"strict_prompt_accuracy": 64.0, "loose_prompt_accuracy": 74.0, "strict_instruction_accuracy": 69.0, "loose_instruction_accuracy": 79.0, "throughput_tok_per_sec": 120.0, "bubble_elimination_pct": 50.0},
                    {"strict_prompt_accuracy": 62.0, "loose_prompt_accuracy": 72.0, "strict_instruction_accuracy": 67.0, "loose_instruction_accuracy": 77.0, "throughput_tok_per_sec": 110.0, "bubble_elimination_pct": 50.0},
                ]

                summary = asyncio.run(
                    run_suite(
                        base_url="http://mock-gw/v1",
                        gateway_url="http://mock-gw",
                        benchmarks="ifeval",
                        concurrency=2,
                        num_runs=3,
                        output_dir=tmpdir,
                    )
                )

                self.assertEqual(summary["num_runs"], 3)
                self.assertIn("aggregated_statistics", summary)
                if_stats = summary["aggregated_statistics"]["ifeval"]
                self.assertEqual(if_stats["strict_prompt_accuracy"]["mean"], 62.0)
                self.assertEqual(if_stats["strict_prompt_accuracy"]["min"], 60.0)
                self.assertEqual(if_stats["strict_prompt_accuracy"]["max"], 64.0)
                self.assertEqual(if_stats["throughput_tok_per_sec"]["mean"], 110.0)
                self.assertTrue(os.path.exists(os.path.join(tmpdir, "eval_report.md")))


if __name__ == "__main__":
    unittest.main()
