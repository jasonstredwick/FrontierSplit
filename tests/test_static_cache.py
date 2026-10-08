"""Tests for StaticCache support in WorkerKVCacheStore and pipeline worker."""

import unittest
from unittest.mock import MagicMock
from frontiersplit.worker import WorkerKVCacheStore


class TestStaticCache(unittest.TestCase):
    def test_worker_kv_cache_store_default_dynamic(self):
        store = WorkerKVCacheStore(use_static_cache=False)
        self.assertFalse(store.use_static_cache)
        cache = store.get_or_create("req-1", prompt_len=10, max_tokens=32)
        # Should allocate aligned capacity
        self.assertIn("req-1", store.sessions)
        self.assertEqual(store.get_seq_len("req-1"), 10)
        store.update_seq_len("req-1", 1)
        self.assertEqual(store.get_seq_len("req-1"), 11)
        store.release("req-1")
        self.assertNotIn("req-1", store.sessions)

    def test_worker_kv_cache_store_static_cache_bucketing(self):
        mock_cfg = MagicMock()
        mock_cfg.num_key_value_heads = 8
        mock_cfg.num_attention_heads = 32
        mock_cfg.hidden_size = 4096

        store = WorkerKVCacheStore(model_config=mock_cfg, use_static_cache=True, device="cpu")
        self.assertTrue(store.use_static_cache)
        # When prompt_len=10, max_tokens=32 -> hard_cap=42 -> bucket should be 64
        cache = store.get_or_create("req-static", prompt_len=10, max_tokens=32)
        self.assertIsNotNone(cache)
        # Should be an instance of StaticCache or fallback
        self.assertIn("req-static", store.sessions)
        self.assertEqual(store.get_seq_len("req-static"), 10)
        store.update_seq_len("req-static", 1)
        self.assertEqual(store.get_seq_len("req-static"), 11)
        store.release("req-static")
        self.assertEqual(store.allocated_tokens, 0)


if __name__ == "__main__":
    unittest.main()
