"""Tests for FrontierSplit Concurrent Request Scheduler and Streaming SSE."""

import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from frontiersplit.gateway import create_gateway_app
from frontiersplit.protocol import GenerationResponse
from frontiersplit.scheduler import PipelineScheduler


class TestConcurrentScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = PipelineScheduler(
            stage0_url="http://node-0:50051",
            num_workers=4,
            total_stages=4,
        )
        self.gateway_app = create_gateway_app(
            stage0_url="http://node-0:50051",
            total_stages=4,
            scheduler=self.scheduler,
        )
        self.client = TestClient(self.gateway_app)

    def test_single_stream_completion(self):
        """Verify standard single-stream completion through scheduler."""
        def mock_post(url, json=None, timeout=None):
            class MockResp:
                status_code = 200
                def json(self):
                    return {
                        "request_id": json["request_id"],
                        "token_id": 101,
                        "text": "frontier ",
                        "is_finished": False,
                        "latency_ms": 2.5,
                        "stage_timings": {"stage_0_compute_ms": 1.0},
                    }
                def raise_for_status(self):
                    pass
            return MockResp()

        with patch("requests.post", side_effect=mock_post):
            req_body = {
                "model": "frontiersplit-mixtral-8x7b",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
                "stream": False,
            }
            resp = self.client.post("/v1/chat/completions", json=req_body)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["object"], "chat.completion")
            self.assertEqual(data["usage"]["completion_tokens"], 4)
            self.assertEqual(data["choices"][0]["message"]["content"], "frontier frontier frontier frontier ")

    def test_streaming_sse_completion(self):
        """Verify Server-Sent Events (SSE) streaming output."""
        def mock_post(url, json=None, timeout=None):
            class MockResp:
                status_code = 200
                def json(self):
                    return {
                        "request_id": json["request_id"],
                        "token_id": 202,
                        "text": "token ",
                        "is_finished": False,
                        "latency_ms": 1.2,
                        "stage_timings": {},
                    }
                def raise_for_status(self):
                    pass
            return MockResp()

        with patch("requests.post", side_effect=mock_post):
            req_body = {
                "model": "frontiersplit-mixtral-8x7b",
                "messages": [{"role": "user", "content": "Stream test"}],
                "max_tokens": 3,
                "stream": True,
            }
            resp = self.client.post("/v1/chat/completions", json=req_body)
            self.assertEqual(resp.status_code, 200)
            self.assertIn("text/event-stream", resp.headers["content-type"])

            lines = resp.text.strip().split("\n\n")
            # Should have 3 token chunks + 1 finish chunk + [DONE]
            self.assertTrue(len(lines) >= 4)
            self.assertEqual(lines[-1], "data: [DONE]")

            first_event = json.loads(lines[0].replace("data: ", ""))
            self.assertEqual(first_event["object"], "chat.completion.chunk")
            self.assertEqual(first_event["choices"][0]["delta"]["content"], "token ")

    def test_concurrent_multi_agent_requests(self):
        """Simulate concurrent multi-agent requests hitting the gateway simultaneously."""
        def mock_post(url, json=None, timeout=None):
            class MockResp:
                status_code = 200
                def json(self):
                    step = json.get("sequence_step", 0)
                    req_id = json.get("request_id", "")
                    return {
                        "request_id": req_id,
                        "token_id": 100 + step,
                        "text": f"ans_{req_id[-4:]} ",
                        "is_finished": False,
                        "latency_ms": 1.5,
                        "stage_timings": {},
                    }
                def raise_for_status(self):
                    pass
            return MockResp()

        concurrency = 8
        tokens_per_req = 4

        with patch("requests.post", side_effect=mock_post):
            async def run_concurrent():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=self.gateway_app),
                    base_url="http://test",
                ) as async_client:
                    async def send_req(i):
                        req_body = {
                            "model": "frontiersplit-mixtral-8x7b",
                            "messages": [{"role": "user", "content": f"Agent query {i}"}],
                            "max_tokens": tokens_per_req,
                            "stream": False,
                        }
                        return await async_client.post("/v1/chat/completions", json=req_body)

                    return await asyncio.gather(*[send_req(i) for i in range(concurrency)])

            results = asyncio.run(run_concurrent())

            for resp in results:
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertEqual(data["usage"]["completion_tokens"], tokens_per_req)

            # Check telemetry metrics recorded
            telem_resp = self.client.get("/v1/telemetry")
            self.assertEqual(telem_resp.status_code, 200)
            telem = telem_resp.json()
            self.assertGreaterEqual(telem["total_submitted"], concurrency)
            self.assertGreaterEqual(telem["total_completed"], concurrency)
            self.assertGreaterEqual(telem["total_tokens_generated"], concurrency * tokens_per_req)

    def test_bubble_factor_calculation(self):
        """Verify theoretical pipeline bubble fraction formula."""
        # When 1 stream active: bubble = (4 - 1) / (1 + 4 - 1) = 3/4 = 0.75
        telem = self.scheduler.get_telemetry()
        self.assertEqual(telem["theoretical_bubble_fraction"], 0.75)
        self.assertEqual(telem["bubble_elimination_pct"], 25.0)


if __name__ == "__main__":
    unittest.main()
