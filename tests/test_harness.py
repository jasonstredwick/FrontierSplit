"""Tests for FrontierSplit Multi-Agent Harness and Telemetry Dashboard."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from benchmarks.agent_harness import MultiAgentHarness
from benchmarks.telemetry_dashboard import render_meter


class TestMultiAgentHarness(unittest.TestCase):
    def setUp(self):
        self.harness = MultiAgentHarness(base_url="http://mock-gateway/v1", model="frontiersplit-test")

    def test_render_meter(self):
        """Test ASCII meter string rendering."""
        bar_50 = render_meter(50.0, width=10)
        self.assertIn("50.0%", bar_50)
        self.assertEqual(bar_50.count("█"), 5)
        self.assertEqual(bar_50.count("░"), 5)

        bar_100 = render_meter(100.0, width=10)
        self.assertIn("100.0%", bar_100)
        self.assertEqual(bar_100.count("█"), 10)

    def test_tree_of_thoughts_execution(self):
        """Test parallel Tree-of-Thoughts reasoning branches."""
        mock_response = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "Thought step"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            request=httpx.Request("POST", "http://mock-gateway/v1/chat/completions"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            res = asyncio.run(self.harness.run_tree_of_thoughts(prompt="Test problem", branches=3, tokens_per_thought=5))

            self.assertEqual(res["mode"], "tree_of_thoughts")
            self.assertEqual(res["branches"], 3)
            self.assertEqual(len(res["branch_results"]), 3)
            # 3 branch calls + 1 synthesis call = 4 total calls
            self.assertEqual(mock_post.call_count, 4)

    def test_agent_swarm_execution(self):
        """Test concurrent multi-agent swarm task execution."""
        mock_response = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-swarm",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "Agent analysis done"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            },
            request=httpx.Request("POST", "http://mock-gateway/v1/chat/completions"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            tasks = ["Task A", "Task B", "Task C", "Task D"]
            res = asyncio.run(self.harness.run_agent_swarm(tasks=tasks, tokens_per_task=8))

            self.assertEqual(res["mode"], "agent_swarm")
            self.assertEqual(res["agent_count"], 4)
            self.assertEqual(len(res["results"]), 4)
            self.assertEqual(mock_post.call_count, 4)

    def test_concurrency_sweep_study(self):
        """Test concurrency sweep benchmarking bubble reduction."""
        mock_response = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-sweep",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "Result tok"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
            },
            request=httpx.Request("POST", "http://mock-gateway/v1/chat/completions"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            sweep = asyncio.run(self.harness.run_concurrency_sweep(concurrency_levels=[1, 4], tokens_per_request=4))

            self.assertEqual(len(sweep), 2)
            # N=1 bubble: (4 - 1)/(1 + 4 - 1) = 3/4 = 0.75
            self.assertEqual(sweep[0]["concurrency_n"], 1)
            self.assertEqual(sweep[0]["bubble_fraction"], 0.75)
            self.assertEqual(sweep[0]["bubble_elimination_pct"], 25.0)

            # N=4 bubble: (4 - 1)/(4 + 4 - 1) = 3/7 = ~0.4286 (57.1% saturated)
            self.assertEqual(sweep[1]["concurrency_n"], 4)
            self.assertAlmostEqual(sweep[1]["bubble_fraction"], 0.4286, places=3)
            self.assertAlmostEqual(sweep[1]["bubble_elimination_pct"], 57.1, places=1)


if __name__ == "__main__":
    unittest.main()
