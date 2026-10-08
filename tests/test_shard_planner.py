"""Unit tests for FrontierSplit Shard Planner."""

import unittest
from unittest.mock import patch
from frontiersplit.shard_planner import ShardPlanner


class TestShardPlanner(unittest.TestCase):
    def test_shard_partitioning_logic(self):
        # Mock config and index
        mock_config = {
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 8,
        }
        mock_index = {
            "metadata": {"total_size": 10 * (1024 ** 3)},
            "weight_map": {
                "model.embed_tokens.weight": "shard-1.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "shard-1.safetensors",
                "model.layers.1.self_attn.q_proj.weight": "shard-2.safetensors",
                "model.layers.2.self_attn.q_proj.weight": "shard-2.safetensors",
                "model.layers.3.self_attn.q_proj.weight": "shard-3.safetensors",
                "model.layers.4.self_attn.q_proj.weight": "shard-3.safetensors",
                "model.layers.5.self_attn.q_proj.weight": "shard-4.safetensors",
                "model.layers.6.self_attn.q_proj.weight": "shard-4.safetensors",
                "model.layers.7.self_attn.q_proj.weight": "shard-5.safetensors",
                "model.norm.weight": "shard-5.safetensors",
                "lm_head.weight": "shard-5.safetensors",
            },
        }

        with patch.object(ShardPlanner, "fetch_config", return_value=mock_config), \
             patch.object(ShardPlanner, "fetch_index", return_value=mock_index):
            plan = ShardPlanner.plan_shards("mock-model", total_stages=4)

            self.assertEqual(plan["total_layers"], 8)
            self.assertEqual(plan["total_stages"], 4)
            self.assertEqual(len(plan["stages"]), 4)

            # Stage 0: layers 0..1 + embeddings -> shard-1, shard-2
            self.assertEqual(plan["stages"][0]["start_layer"], 0)
            self.assertEqual(plan["stages"][0]["end_layer"], 1)
            self.assertEqual(plan["stages"][0]["needed_shards"], ["shard-1.safetensors", "shard-2.safetensors"])

            # Stage 3 (Final): layers 6..7 + norm + lm_head -> shard-4, shard-5
            self.assertEqual(plan["stages"][3]["start_layer"], 6)
            self.assertEqual(plan["stages"][3]["end_layer"], 7)
            self.assertEqual(plan["stages"][3]["needed_shards"], ["shard-4.safetensors", "shard-5.safetensors"])


if __name__ == "__main__":
    unittest.main()
