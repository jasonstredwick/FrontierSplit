"""FrontierSplit Pipeline Worker.

Runs on each compute node in the cluster, hosting assigned transformer layers.
Receives activation packets, computes local layer math, and forwards the resulting
tensor to the next node in the pipeline chain. Supports both real Hugging Face PyTorch
transformer layers and lightweight synthetic layers for testing.
"""

from __future__ import annotations

import argparse
import gc
import time
from typing import Any, List, Optional
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
import uvicorn

from frontiersplit.protocol import ActivationPacket, GenerationResponse

# Optional PyTorch and Hugging Face imports
try:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def create_worker_app(
    stage_id: int,
    total_stages: int,
    downstream_url: Optional[str] = None,
    model_name_or_path: Optional[str] = None,
    device: Optional[str] = None,
    hidden_size: int = 4096,
    vocab_size: int = 32000,
) -> FastAPI:
    """Creates a FastAPI app representing a single pipeline stage worker."""
    app = FastAPI(title=f"FrontierSplit-Worker-Stage-{stage_id}")
    is_first_stage = (stage_id == 0)
    is_final_stage = (stage_id == total_stages - 1)

    use_real_model = bool(model_name_or_path and HAS_TORCH)
    tokenizer = None
    assigned_layers: List[Any] = []
    embed_tokens = None
    final_norm = None
    lm_head = None
    layer_range = (0, 0)
    resolved_device = "cpu"

    if use_real_model:
        resolved_device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        print(f"[Worker Stage {stage_id}] Loading real model weights: {model_name_or_path} on device: {resolved_device}...")
        
        config = AutoConfig.from_pretrained(model_name_or_path)
        total_layers = getattr(config, "num_hidden_layers", 32)
        layers_per_stage = total_layers // total_stages
        remainder = total_layers % total_stages

        start_layer = stage_id * layers_per_stage + min(stage_id, remainder)
        assigned_count = layers_per_stage + (1 if stage_id < remainder else 0)
        end_layer = start_layer + assigned_count - 1
        layer_range = (start_layer, end_layer)
        print(f"[Worker Stage {stage_id}] Assigned layers {start_layer} to {end_layer} ({assigned_count} layers total).")

        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        except Exception as e:
            print(f"[Worker Stage {stage_id}] Warning: Failed to load tokenizer: {e}")

        # Load model weights in FP16
        full_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        base_model = getattr(full_model, "model", getattr(full_model, "transformer", full_model))
        all_layers = getattr(base_model, "layers", getattr(base_model, "h", []))

        if is_first_stage:
            raw_embed = getattr(base_model, "embed_tokens", getattr(base_model, "wte", None))
            embed_tokens = raw_embed.to(resolved_device) if raw_embed is not None else None

        for idx in range(start_layer, end_layer + 1):
            assigned_layers.append(all_layers[idx].to(resolved_device))

        if is_final_stage:
            raw_norm = getattr(base_model, "norm", getattr(base_model, "ln_f", None))
            final_norm = raw_norm.to(resolved_device) if raw_norm is not None else None
            raw_lm_head = getattr(full_model, "lm_head", None)
            lm_head = raw_lm_head.to(resolved_device) if raw_lm_head is not None else None

        # Free unassigned layers and garbage collect
        del full_model
        del base_model
        del all_layers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[Worker Stage {stage_id}] Successfully mounted assigned layers into VRAM.")

    else:
        # Synthetic fallback mode for testing and development
        np.random.seed(42 + stage_id)
        layer_proj = np.eye(hidden_size, dtype=np.float32) + 0.001 * np.random.randn(hidden_size, hidden_size).astype(np.float32)
        embeddings = np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.02 if is_first_stage else None
        lm_head_synth = np.random.randn(hidden_size, vocab_size).astype(np.float32) * 0.02 if is_final_stage else None

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "stage_id": stage_id,
            "total_stages": total_stages,
            "is_first_stage": is_first_stage,
            "is_final_stage": is_final_stage,
            "downstream_url": downstream_url,
            "use_real_model": use_real_model,
            "device": resolved_device,
            "layer_range": layer_range,
        }

    @app.post("/forward", response_model=GenerationResponse)
    def forward(packet: ActivationPacket):
        start_time = time.time()

        if use_real_model:
            with torch.no_grad():
                if is_first_stage and packet.tokens:
                    token_tensor = torch.tensor([packet.tokens], device=resolved_device, dtype=torch.long)
                    hidden_states = embed_tokens(token_tensor)
                else:
                    arr = packet.get_tensor()
                    hidden_states = torch.from_numpy(arr).to(device=resolved_device, dtype=torch.float16)

                # Forward through assigned transformer layers
                for layer in assigned_layers:
                    layer_out = layer(hidden_states)
                    hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

                stage_compute_ms = (time.time() - start_time) * 1000
                timings = dict(packet.stage_timings)
                timings[f"stage_{stage_id}_compute_ms"] = stage_compute_ms

                if not is_final_stage:
                    if not downstream_url:
                        raise HTTPException(status_code=500, detail="Missing downstream_url for intermediate stage")

                    next_act = hidden_states.detach().cpu().to(torch.float32).numpy()
                    next_packet = ActivationPacket(
                        request_id=packet.request_id,
                        sequence_step=packet.sequence_step,
                        stage_id=stage_id + 1,
                        is_prefill=packet.is_prefill,
                        stage_timings=timings,
                    )
                    next_packet.set_tensor(next_act)

                    try:
                        resp = requests.post(f"{downstream_url}/forward", json=next_packet.model_dump(), timeout=30)
                        resp.raise_for_status()
                        return resp.json()
                    except Exception as e:
                        raise HTTPException(status_code=502, detail=f"Downstream handoff to {downstream_url} failed: {e}")

                else:
                    # Final Stage: Compute LM head logits and select next token
                    if final_norm is not None:
                        hidden_states = final_norm(hidden_states)

                    last_token_hidden = hidden_states[:, -1, :]  # shape: [1, hidden_size]
                    logits = lm_head(last_token_hidden)  # shape: [1, vocab_size]

                    next_token_id = int(torch.argmax(logits, dim=-1).item())
                    elapsed_ms = (time.time() - start_time) * 1000

                    if tokenizer is not None:
                        decoded_text = tokenizer.decode([next_token_id])
                        eos_id = getattr(tokenizer, "eos_token_id", None)
                        is_finished = (next_token_id == eos_id) or (packet.sequence_step >= 256)
                    else:
                        decoded_text = f"tok_{next_token_id % 100} "
                        is_finished = (packet.sequence_step >= 32)

                    return GenerationResponse(
                        request_id=packet.request_id,
                        token_id=next_token_id,
                        text=decoded_text,
                        is_finished=is_finished,
                        latency_ms=elapsed_ms,
                        stage_timings=timings,
                    )

        else:
            # Synthetic matrix compute for testing
            if is_first_stage and packet.tokens:
                token_ids = packet.tokens
                activation = embeddings[token_ids]
                if activation.ndim == 2:
                    activation = activation[np.newaxis, ...]
            else:
                activation = packet.get_tensor()

            activation = np.matmul(activation, layer_proj)
            stage_compute_ms = (time.time() - start_time) * 1000
            timings = dict(packet.stage_timings)
            timings[f"stage_{stage_id}_compute_ms"] = stage_compute_ms

            if not is_final_stage:
                if not downstream_url:
                    raise HTTPException(status_code=500, detail="Missing downstream_url for intermediate stage")

                next_packet = ActivationPacket(
                    request_id=packet.request_id,
                    sequence_step=packet.sequence_step,
                    stage_id=stage_id + 1,
                    is_prefill=packet.is_prefill,
                    stage_timings=timings,
                )
                next_packet.set_tensor(activation)

                try:
                    resp = requests.post(f"{downstream_url}/forward", json=next_packet.model_dump(), timeout=30)
                    resp.raise_for_status()
                    return resp.json()
                except Exception as e:
                    raise HTTPException(status_code=502, detail=f"Downstream handoff to {downstream_url} failed: {e}")

            else:
                last_token_hidden = activation[0, -1, :]
                logits = np.matmul(last_token_hidden, lm_head_synth)
                next_token_id = int(np.argmax(logits))
                elapsed_ms = (time.time() - start_time) * 1000
                mock_word = f"tok_{next_token_id % 100} "
                return GenerationResponse(
                    request_id=packet.request_id,
                    token_id=next_token_id,
                    text=mock_word,
                    is_finished=(packet.sequence_step >= 32),
                    latency_ms=elapsed_ms,
                    stage_timings=timings,
                )

    return app


def main():
    parser = argparse.ArgumentParser(description="FrontierSplit Pipeline Stage Worker")
    parser.add_argument("--stage-id", type=int, required=True, help="Stage index (0 to total_stages - 1)")
    parser.add_argument("--total-stages", type=int, default=2, help="Total pipeline stages")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on")
    parser.add_argument("--downstream-url", type=str, default=None, help="Downstream worker URL")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    parser.add_argument("--model-name", type=str, default=None, help="Hugging Face model ID to load")
    parser.add_argument("--device", type=str, default=None, help="PyTorch device (cuda, mps, cpu)")
    args = parser.parse_args()

    app = create_worker_app(
        stage_id=args.stage_id,
        total_stages=args.total_stages,
        downstream_url=args.downstream_url,
        model_name_or_path=args.model_name,
        device=args.device,
    )
    print(f"Starting FrontierSplit Worker Stage {args.stage_id}/{args.total_stages} on port {args.port}...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
