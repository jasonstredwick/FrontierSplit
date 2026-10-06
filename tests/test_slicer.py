"""Unit tests for ModelSlicer."""

import unittest
from frontiersplit.slicer import ModelSlicer


class TestModelSlicer(unittest.TestCase):
    def test_synthetic_moe_profiling(self):
        # Synthetic config matching standard MoE parameters
        config = {
            "architectures": ["MixtralForCausalLM"],
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "vocab_size": 32000,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "intermediate_size": 14336,
        }

        profile_fp16 = ModelSlicer.profile_model(config, model_id="test-moe", precision="fp16")
        self.assertEqual(profile_fp16.total_layers, 32)
        self.assertEqual(profile_fp16.hidden_size, 4096)
        self.assertEqual(profile_fp16.bytes_per_param, 2.0)
        self.assertEqual(profile_fp16.activation_bytes_per_token, 4096 * 2)  # 8 KB

        # Test partition feasibility on 4 vs 5 nodes
        plan_4node = ModelSlicer.plan_cluster_partition(profile_fp16, num_nodes=4, vram_per_node_gb=24.0)
        # In FP16, 86.99 GB / 4 = ~21.7GB, which exceeds 85% safety threshold (20.4 GB)
        self.assertFalse(plan_4node.is_feasible)

        plan_5node = ModelSlicer.plan_cluster_partition(profile_fp16, num_nodes=5, vram_per_node_gb=24.0)
        self.assertTrue(plan_5node.is_feasible)

        # In FP8, 4 nodes should be comfortably feasible
        profile_fp8 = ModelSlicer.profile_model(config, model_id="test-moe", precision="fp8")
        self.assertEqual(profile_fp8.bytes_per_param, 1.0)
        self.assertEqual(profile_fp8.activation_bytes_per_token, 4096 * 1)  # 4 KB
        plan_4node_fp8 = ModelSlicer.plan_cluster_partition(profile_fp8, num_nodes=4, vram_per_node_gb=24.0)
        self.assertTrue(plan_4node_fp8.is_feasible)


if __name__ == "__main__":
    unittest.main()
