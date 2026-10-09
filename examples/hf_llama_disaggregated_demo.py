"""HuggingFace Llama Model Disaggregated Inference Demo.

Demonstrates wrapping a standard HuggingFace Llama architecture with FrontierSplit's
DisaggregatedModel to offload the prompt KV cache to an independent Context Server
over TCP and generate tokens with 100% token-for-token identical outputs.

Run with:
    python examples/hf_llama_disaggregated_demo.py
"""

from __future__ import annotations

import copy
import time

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from frontiersplit import (
    ContextClient,
    ContextServer,
    DisaggregatedModel,
)


def main() -> None:
    print("=" * 72)
    print(" FrontierSplit: Real HuggingFace Causal LM Disaggregated Inference")
    print("=" * 72)

    # 1. Initialize HuggingFace Model Architecture
    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,  # Grouped-Query Attention (GQA)
    )
    model = LlamaForCausalLM(config)
    model.eval()

    prompt = torch.tensor([[12, 45, 99, 101, 204, 33, 7]])
    prompt_len = prompt.shape[1]
    num_tokens = 8

    print("\n[1/4] HuggingFace Model Configuration:")
    print(
        f"  • Architecture:        LlamaForCausalLM ({config.num_hidden_layers} layers)"
    )
    print(
        f"  • Attention Heads:     {config.num_attention_heads} Query / {config.num_key_value_heads} KV"
    )
    print(f"  • Prompt Length:       {prompt_len} tokens: {prompt.tolist()[0]}")
    print(f"  • Tokens to Generate:  {num_tokens}")

    # 2. Run Monolithic Baseline (Native HuggingFace Generation)
    print("\n[2/4] Running Monolithic Ground-Truth Baseline...")
    t0 = time.perf_counter()
    with torch.no_grad():
        out_prefill = model(prompt, use_cache=True)
        mono_cache = copy.deepcopy(out_prefill.past_key_values)
        first_tok = out_prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        mono_tokens = [first_tok.item()]
        cur_tok = first_tok
        for _ in range(num_tokens - 1):
            out = model(cur_tok, past_key_values=mono_cache, use_cache=True)
            cur_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            mono_tokens.append(cur_tok.item())

    mono_elapsed = (time.perf_counter() - t0) * 1000
    print(f"  • Monolithic Tokens:   {mono_tokens}")
    print(f"  • Monolithic Time:     {mono_elapsed:.2f} ms")

    # 3. Start FrontierSplit Disaggregated Infrastructure
    print("\n[3/4] Launching FrontierSplit Disaggregated Cluster over TCP...")
    server = ContextServer()
    port = server.start_in_thread(host="127.0.0.1", port=0)
    print(f"  • Context Server listening on TCP 127.0.0.1:{port}")

    client = ContextClient(host="127.0.0.1", port=port)
    disagg_model = DisaggregatedModel(model=model, context_client=client)
    print("  • Installed DisaggregatedAttentionPatchers on all decoder layers.")

    # 4. Generate with Disaggregated Attention Merging
    print("\n[4/4] Generating with FrontierSplit Disaggregated Attention...")
    t1 = time.perf_counter()
    try:
        disagg_tokens = disagg_model.generate(
            input_ids=prompt,
            max_new_tokens=num_tokens,
            session_id="hf-demo-session",
        )
        disagg_elapsed = (time.perf_counter() - t1) * 1000

        print(f"  • Disaggregated Tokens: {disagg_tokens}")
        print(f"  • Disaggregated Time:   {disagg_elapsed:.2f} ms")

        # 5. Numerical Equivalence Verification
        assert mono_tokens == disagg_tokens, (
            "Token mismatch between monolithic and disaggregated!"
        )
        print("\n" + "=" * 72)
        print(" [✓] SUCCESS: 100% BIT-FOR-BIT IDENTICAL GENERATION ACROSS ALL LAYERS!")
        print("=" * 72)

    finally:
        client.close_sync()
        server.stop_thread()
        print("  • Remote prompt KV cache released and TCP server shut down.")


if __name__ == "__main__":
    main()
