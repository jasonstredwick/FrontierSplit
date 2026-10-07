"""FrontierSplit Pipeline Worker.

Runs on each compute node in the cluster, hosting assigned transformer layers.
Receives activation packets, computes local layer math, and forwards the resulting
tensor to the next node in the pipeline chain.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
import uvicorn

from frontiersplit.protocol import ActivationPacket, GenerationResponse


def create_worker_app(
    stage_id: int,
    total_stages: int,
    downstream_url: Optional[str] = None,
    hidden_size: int = 4096,
    vocab_size: int = 32000,
) -> FastAPI:
    """Creates a FastAPI app representing a single pipeline stage worker."""
    app = FastAPI(title=f"FrontierSplit-Worker-Stage-{stage_id}")
    is_first_stage = (stage_id == 0)
    is_final_stage = (stage_id == total_stages - 1)

    # Initialize synthetic weight matrices for testing / development
    np.random.seed(42 + stage_id)
    # Layer projection weights (simulating layer forward transformation)
    layer_proj = np.eye(hidden_size, dtype=np.float32) + 0.001 * np.random.randn(hidden_size, hidden_size).astype(np.float32)

    if is_first_stage:
        # Embedding table
        embeddings = np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.02
    else:
        embeddings = None

    if is_final_stage:
        # LM Head projection
        lm_head = np.random.randn(hidden_size, vocab_size).astype(np.float32) * 0.02
    else:
        lm_head = None

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "stage_id": stage_id,
            "total_stages": total_stages,
            "is_first_stage": is_first_stage,
            "is_final_stage": is_final_stage,
            "downstream_url": downstream_url,
        }

    @app.post("/forward", response_model=GenerationResponse)
    def forward(packet: ActivationPacket):
        start_time = time.time()

        if is_first_stage and packet.tokens:
            # Stage 0: Look up embeddings for incoming tokens
            token_ids = packet.tokens
            activation = embeddings[token_ids]  # shape: [seq_len, hidden_size]
            if activation.ndim == 2:
                activation = activation[np.newaxis, ...]  # shape: [1, seq_len, hidden_size]
        else:
            # Intermediate or final stage: deserialize incoming activation
            activation = packet.get_tensor()

        # Compute assigned transformer layers (layer projection + activation)
        # In real deployment, this invokes the assigned PyTorch/CUDA MoE blocks
        activation = np.matmul(activation, layer_proj)
        stage_compute_ms = (time.time() - start_time) * 1000
        timings = dict(packet.stage_timings)
        timings[f"stage_{stage_id}_compute_ms"] = stage_compute_ms

        if not is_final_stage:
            # Forward activation downstream to next node in the pipeline
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
            # Final Stage: Compute LM head logits and select next token
            last_token_hidden = activation[0, -1, :]  # shape: [hidden_size]
            logits = np.matmul(last_token_hidden, lm_head)  # shape: [vocab_size]

            # Greedy or argmax sampling
            next_token_id = int(np.argmax(logits))
            elapsed_ms = (time.time() - start_time) * 1000

            # Mock token decode mapping for testing
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
    parser.add_argument("--total-stages", type=int, default=4, help="Total pipeline stages")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on")
    parser.add_argument("--downstream-url", type=str, default=None, help="Downstream worker URL (e.g. http://node-1:50051)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    args = parser.parse_args()

    app = create_worker_app(
        stage_id=args.stage_id,
        total_stages=args.total_stages,
        downstream_url=args.downstream_url,
    )
    print(f"Starting FrontierSplit Worker Stage {args.stage_id}/{args.total_stages} on port {args.port}...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
