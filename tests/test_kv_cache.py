"""Unit tests for FrontierSplit Dynamic Contiguous KV Cache & Session Store."""

import unittest
from fastapi.testclient import TestClient

from frontiersplit.protocol import (
    ActivationPacket,
    BatchedActivationPacket,
    ReleaseSessionPacket,
    ReleaseSessionResponse,
)
from frontiersplit.worker import WorkerKVCacheStore, create_worker_app


class TestWorkerKVCacheStore(unittest.TestCase):
    """Test suite verifying tile-aligned capacity sizing and session tracking."""

    def setUp(self):
        self.store = WorkerKVCacheStore(max_tokens_budget=10_000)

    def test_tile_aligned_small_cap(self):
        """Rule 1: hard_cap <= 256 must allocate strictly to ceil_16(hard_cap)."""
        # prompt=30, max_tokens=20 => hard_cap=50 => ceil_16(50) = 64
        cache = self.store.get_or_create("req_small_1", prompt_len=30, max_tokens=20)
        self.assertIsNotNone(cache)
        self.assertEqual(self.store.session_caps["req_small_1"], 64)
        self.assertEqual(self.store.allocated_tokens, 64)

    def test_tile_aligned_boundary(self):
        """Check exact multiples of 16."""
        # prompt=64, max_tokens=16 => hard_cap=80 => ceil_16(80) = 80
        self.store.get_or_create("req_exact_1", prompt_len=64, max_tokens=16)
        self.assertEqual(self.store.session_caps["req_exact_1"], 80)
        self.assertEqual(self.store.allocated_tokens, 80)

    def test_session_lifecycle_and_reclamation(self):
        """Verify get, update, and release of session tokens."""
        self.store.get_or_create("req_life_1", prompt_len=100, max_tokens=100)
        self.assertEqual(self.store.get_seq_len("req_life_1"), 100)

        # Advance step
        new_len = self.store.update_seq_len("req_life_1", 1)
        self.assertEqual(new_len, 101)
        self.assertEqual(self.store.get_seq_len("req_life_1"), 101)

        # Release session
        prev_alloc = self.store.allocated_tokens
        released = self.store.release("req_life_1")
        self.assertTrue(released)
        self.assertIsNone(self.store.get("req_life_1"))
        self.assertLess(self.store.allocated_tokens, prev_alloc)
        self.assertEqual(self.store.allocated_tokens, 0)


class TestWorkerKVCacheEndpoints(unittest.TestCase):
    """Test suite verifying HTTP /forward and /release_sessions endpoints on worker."""

    def setUp(self):
        self.app = create_worker_app(
            stage_id=0,
            total_stages=1,
            hidden_size=64,
            vocab_size=100,
        )
        self.client = TestClient(self.app)

    def test_forward_and_release_lifecycle(self):
        """Prefill -> decode -> release flow."""
        # 1. Check initial health
        health = self.client.get("/health").json()
        self.assertEqual(health["active_sessions"], 0)

        # 2. Prefill forward
        prefill_pkt = ActivationPacket(
            request_id="test_req_e2e_1",
            sequence_step=0,
            stage_id=0,
            is_prefill=True,
            use_kv_cache=True,
            max_tokens=32,
            tokens=[1, 2, 3, 4],
        )
        resp = self.client.post("/forward", json=prefill_pkt.model_dump())
        self.assertEqual(resp.status_code, 200)

        # Verify active session registered
        health = self.client.get("/health").json()
        self.assertEqual(health["active_sessions"], 1)

        # 3. Decode step
        decode_pkt = ActivationPacket(
            request_id="test_req_e2e_1",
            sequence_step=1,
            stage_id=0,
            is_prefill=False,
            use_kv_cache=True,
            max_tokens=32,
            tokens=[5],
        )
        resp2 = self.client.post("/forward", json=decode_pkt.model_dump())
        self.assertEqual(resp2.status_code, 200)

        # 4. Release sessions
        rel_pkt = ReleaseSessionPacket(request_ids=["test_req_e2e_1"])
        rel_resp = self.client.post("/release_sessions", json=rel_pkt.model_dump())
        self.assertEqual(rel_resp.status_code, 200)
        data = rel_resp.json()
        self.assertEqual(data["released_count"], 1)
        self.assertEqual(data["active_sessions"], 0)

        # Verify health shows 0 active sessions
        health_after = self.client.get("/health").json()
        self.assertEqual(health_after["active_sessions"], 0)
        self.assertEqual(health_after["allocated_kv_tokens"], 0)

    def test_batched_gemm_decode_lifecycle(self):
        """Verify batched prefill followed by batched GEMM decode across multiple streams."""
        # 1. Batched prefill for 2 streams
        prefill_pkt = BatchedActivationPacket(
            request_ids=["bat_req_1", "bat_req_2"],
            sequence_steps=[0, 0],
            stage_id=0,
            is_prefill=True,
            use_kv_cache=True,
            tokens_batch=[[1, 2, 3], [4, 5, 6, 7]],
        )
        resp1 = self.client.post("/forward_batched", json=prefill_pkt.model_dump())
        self.assertEqual(resp1.status_code, 200)

        health = self.client.get("/health").json()
        self.assertEqual(health["active_sessions"], 2)

        # 2. Batched decode step (single token per stream)
        decode_pkt = BatchedActivationPacket(
            request_ids=["bat_req_1", "bat_req_2"],
            sequence_steps=[1, 1],
            stage_id=0,
            is_prefill=False,
            use_kv_cache=True,
            tokens_batch=[[10], [20]],
        )
        resp2 = self.client.post("/forward_batched", json=decode_pkt.model_dump())
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()
        self.assertEqual(len(data2["responses"]), 2)

        # 3. Clean release
        rel_pkt = ReleaseSessionPacket(request_ids=["bat_req_1", "bat_req_2"])
        rel_resp = self.client.post("/release_sessions", json=rel_pkt.model_dump())
        self.assertEqual(rel_resp.status_code, 200)
        self.assertEqual(rel_resp.json()["released_count"], 2)

        health_after = self.client.get("/health").json()
        self.assertEqual(health_after["active_sessions"], 0)


if __name__ == "__main__":
    unittest.main()
