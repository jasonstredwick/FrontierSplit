"""Tests for FrontierSplit SWE-bench Lite Compound Agent Harness."""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
import httpx

from benchmarks.eval_swebench import (
    PatchVerificationResult,
    SWEBenchAgentRunner,
    SWEBenchInstance,
    extract_unified_diff,
    load_swebench_from_json,
    verify_patch_syntax,
)


class TestSWEBenchDiffParser(unittest.TestCase):
    def test_extract_unified_diff_from_markdown(self):
        markdown_text = (
            "Here is the proposed fix for the issue:\n\n"
            "```diff\n"
            "--- a/django/contrib/auth/validators.py\n"
            "+++ b/django/contrib/auth/validators.py\n"
            "@@ -17,3 +17,3 @@ class ASCIIUsernameValidator:\n"
            "-    regex = r'^[\\w.@+-]+$'\n"
            "+    regex = r'\\A[\\w.@+-]+\\Z'\n"
            "```\n"
            "This fixes the trailing newline bug."
        )
        diff = extract_unified_diff(markdown_text)
        self.assertTrue(diff.startswith("--- a/django/contrib/auth/validators.py"))
        self.assertTrue(diff.endswith("+    regex = r'\\A[\\w.@+-]+\\Z'"))

    def test_verify_patch_syntax_valid(self):
        patch = (
            "diff --git a/requests/models.py b/requests/models.py\n"
            "--- a/requests/models.py\n"
            "+++ b/requests/models.py\n"
            "@@ -638,2 +638,3 @@ class Response(object):\n"
            "-        except socket.error:\n"
            "+        except (socket.error, socket.timeout):\n"
        )
        res = verify_patch_syntax(patch, target_files=["requests/models.py"])
        self.assertTrue(res.is_valid_diff)
        self.assertEqual(res.hunk_count, 1)
        self.assertEqual(res.lines_added, 1)
        self.assertEqual(res.lines_removed, 1)
        self.assertIn("requests/models.py", res.affected_files)
        self.assertTrue(res.matches_target_files)

    def test_verify_patch_syntax_invalid(self):
        bad_patch = "I changed the regex from ^ to \\A, but I didn't write diff format."
        res = verify_patch_syntax(bad_patch)
        self.assertFalse(res.is_valid_diff)
        self.assertEqual(res.hunk_count, 0)
        self.assertIsNotNone(res.error_message)


class TestSWEBenchAgentRunner(unittest.TestCase):
    def setUp(self):
        self.runner = SWEBenchAgentRunner(
            base_url="http://mock-gateway/v1",
            model="frontiersplit-test",
            concurrency=2,
        )

    def test_run_compound_agent_instance_mocked(self):
        # Step 1 response (analysis) followed by Step 2 response (patch diff)
        analysis_resp = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-step1",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "Root cause identified in validators.py."}}],
                "usage": {"completion_tokens": 15},
            },
            request=httpx.Request("POST", "http://mock-gateway/v1/chat/completions"),
        )
        patch_resp = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-step2",
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "```diff\n"
                                "--- a/django/contrib/auth/validators.py\n"
                                "+++ b/django/contrib/auth/validators.py\n"
                                "@@ -1,2 +1,2 @@\n"
                                "- old\n"
                                "+ new\n"
                                "```"
                            ),
                        }
                    }
                ],
                "usage": {"completion_tokens": 25},
            },
            request=httpx.Request("POST", "http://mock-gateway/v1/chat/completions"),
        )

        instance = SWEBenchInstance(
            instance_id="django-test-1",
            repo="django/django",
            problem_statement="Regex issue in validators",
            target_files=["django/contrib/auth/validators.py"],
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [analysis_resp, patch_resp]
            async def run_test():
                async with httpx.AsyncClient() as client:
                    return await self.runner.run_compound_agent_instance(client, instance, tokens_per_step=16)

            res = asyncio.run(run_test())

            self.assertEqual(res["instance_id"], "django-test-1")
            self.assertEqual(res["total_tokens"], 40)
            self.assertEqual(len(res["step_metrics"]), 2)
            self.assertTrue(res["is_valid_diff"])
            self.assertTrue(res["matches_target_files"])
            self.assertEqual(mock_post.call_count, 2)

    def test_run_evaluation_summary(self):
        step_resp = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-eval",
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "--- a/src/flask/blueprints.py\n+++ b/src/flask/blueprints.py\n@@ -1,1 +1,2 @@\n+ check\n",
                        }
                    }
                ],
                "usage": {"completion_tokens": 20},
            },
            request=httpx.Request("POST", "http://mock-gateway/v1/chat/completions"),
        )

        test_instances = [
            SWEBenchInstance(
                instance_id="flask-test",
                repo="pallets/flask",
                problem_statement="Validate blueprint name",
                target_files=["src/flask/blueprints.py"],
            )
        ]

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = step_resp
            summary = asyncio.run(self.runner.run_evaluation(dataset=test_instances, tokens_per_step=10))

            self.assertEqual(summary["total_instances"], 1)
            self.assertEqual(summary["valid_diff_count"], 1)
            self.assertEqual(summary["valid_diff_rate_pct"], 100.0)
            self.assertEqual(summary["target_files_matched_pct"], 100.0)
            self.assertIn("bubble_fraction", summary)
            self.assertIn("bubble_elimination_pct", summary)

    def test_load_swebench_from_json(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tf:
            json.dump([
                {
                    "instance_id": "test__inst-1",
                    "repo": "test/repo",
                    "problem_statement": "Bug description",
                    "target_files": ["module.py"],
                }
            ], tf)
            temp_path = tf.name

        try:
            loaded = load_swebench_from_json(temp_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].instance_id, "test__inst-1")
            self.assertEqual(loaded[0].repo, "test/repo")
            self.assertEqual(loaded[0].target_files, ["module.py"])
        finally:
            os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
