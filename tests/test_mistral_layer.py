"""Unit test for isolated Mistral transformer layers."""

import unittest

class TestMistralLayerInterface(unittest.TestCase):
    def test_decoder_layer_signature(self):
        try:
            import torch
            from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralConfig
        except ImportError:
            self.skipTest("torch or transformers not installed")

        config = MistralConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
        )
        layer = MistralDecoderLayer(config, layer_idx=0)
        x = torch.randn(1, 4, 128)
        
        # Test calling layer directly with just hidden_states
        import traceback
        try:
            from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding
            rotary_emb = MistralRotaryEmbedding(config=config)
            seq_len = x.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
            pos_emb = rotary_emb(x, position_ids)
            print("pos_emb shape/type:", type(pos_emb), [p.shape for p in pos_emb] if isinstance(pos_emb, tuple) else pos_emb)
            
            # Layer 0
            out0 = layer(x, position_embeddings=pos_emb)
            h0 = out0[0] if isinstance(out0, tuple) else out0
            print("Layer 0 output shape:", h0.shape)

            # Layer 1
            layer1 = MistralDecoderLayer(config, layer_idx=1)
            out1 = layer1(h0, position_embeddings=pos_emb)
            h1 = out1[0] if isinstance(out1, tuple) else out1
            print("Layer 1 output shape:", h1.shape)
            self.assertEqual(h1.shape, x.shape)
        except Exception:
            traceback.print_exc()
            self.fail("Multi-layer forward failed")

if __name__ == "__main__":
    unittest.main()
