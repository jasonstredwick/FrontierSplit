"""End-to-end tests for FrontierSplit Pipeline & Gateway."""

import unittest
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from frontiersplit.protocol import ActivationPacket
from frontiersplit.worker import create_worker_app
from frontiersplit.gateway import create_gateway_app


class TestPipelineFlow(unittest.TestCase):
    def setUp(self):
        # Create a 4-stage pipeline locally
        self.stage3_app = create_worker_app(stage_id=3, total_stages=4, downstream_url=None, hidden_size=64, vocab_size=256)
        self.client_stage3 = TestClient(self.stage3_app)

        self.stage2_app = create_worker_app(stage_id=2, total_stages=4, downstream_url="http://node-3:50051", hidden_size=64, vocab_size=256)
        self.client_stage2 = TestClient(self.stage2_app)

        self.stage1_app = create_worker_app(stage_id=1, total_stages=4, downstream_url="http://node-2:50051", hidden_size=64, vocab_size=256)
        self.client_stage1 = TestClient(self.stage1_app)

        self.stage0_app = create_worker_app(stage_id=0, total_stages=4, downstream_url="http://node-1:50051", hidden_size=64, vocab_size=256)
        self.client_stage0 = TestClient(self.stage0_app)

        self.gateway_app = create_gateway_app(stage0_url="http://node-0:50051")
        self.client_gateway = TestClient(self.gateway_app)

    def test_direct_stage3_execution(self):
        """Verify final stage produces token logits and generation response."""
        packet = ActivationPacket(
            request_id="req-test-1",
            sequence_step=1,
            stage_id=3,
            is_prefill=False,
        )
        synthetic_act = np.random.randn(1, 4, 64).astype(np.float32)
        packet.set_tensor(synthetic_act)

        resp = self.client_stage3.post("/forward", json=packet.model_dump())
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("token_id", data)
        self.assertIn("text", data)
        self.assertIn("latency_ms", data)
        self.assertIsInstance(data["token_id"], int)

    def test_full_4stage_pipeline_forwarding(self):
        """Verify 4-stage chained forwarding from Stage 0 through Stage 3."""
        # Intercept requests.post to route downstream calls to the appropriate TestClient
        def mock_post(url, json=None, timeout=None):
            class MockResponse:
                def __init__(self, res):
                    self._res = res
                    self.status_code = res.status_code
                def json(self):
                    return self._res.json()
                def raise_for_status(self):
                    if self.status_code >= 400:
                        raise RuntimeError(f"HTTP {self.status_code}")

            if "node-1" in url:
                return MockResponse(self.client_stage1.post("/forward", json=json))
            elif "node-2" in url:
                return MockResponse(self.client_stage2.post("/forward", json=json))
            elif "node-3" in url:
                return MockResponse(self.client_stage3.post("/forward", json=json))
            raise ValueError(f"Unexpected mock URL: {url}")

        with patch("requests.post", side_effect=mock_post):
            initial_packet = ActivationPacket(
                request_id="req-chain-1",
                sequence_step=0,
                stage_id=0,
                is_prefill=True,
                tokens=[10, 20, 30],
            )
            resp = self.client_stage0.post("/forward", json=initial_packet.model_dump())
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("token_id", data)
            self.assertIn("text", data)
            self.assertEqual(data["request_id"], "req-chain-1")

    def test_gateway_chat_completion(self):
        """Verify Gateway /v1/chat/completions end-to-end request."""
        def mock_post(url, json=None, timeout=None):
            class MockResponse:
                def __init__(self, res):
                    self._res = res
                    self.status_code = res.status_code
                def json(self):
                    return self._res.json()
                def raise_for_status(self):
                    if self.status_code >= 400:
                        raise RuntimeError(f"HTTP {self.status_code}")

            if "node-0" in url:
                return MockResponse(self.client_stage0.post("/forward", json=json))
            elif "node-1" in url:
                return MockResponse(self.client_stage1.post("/forward", json=json))
            elif "node-2" in url:
                return MockResponse(self.client_stage2.post("/forward", json=json))
            elif "node-3" in url:
                return MockResponse(self.client_stage3.post("/forward", json=json))
            raise ValueError(f"Unexpected mock URL: {url}")

        with patch("requests.post", side_effect=mock_post):
            req_body = {
                "model": "frontiersplit-mixtral-8x7b",
                "messages": [
                    {"role": "user", "content": "Explain pipeline parallelism in MoE"}
                ],
                "max_tokens": 5,
            }
            resp = self.client_gateway.post("/v1/chat/completions", json=req_body)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["object"], "chat.completion")
            self.assertEqual(len(data["choices"]), 1)
            self.assertEqual(data["choices"][0]["message"]["role"], "assistant")
            self.assertTrue(len(data["choices"][0]["message"]["content"]) > 0)
            self.assertEqual(data["usage"]["completion_tokens"], 5)


if __name__ == "__main__":
    unittest.main()
