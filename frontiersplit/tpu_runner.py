"""FrontierSplit Cloud TPU Multi-Core Pipeline Runner.

Spawns all 8 pipeline stage workers across the 8 TPU v5e cores on a Cloud TPU VM
using torch_xla.distributed.xla_multiprocessing.spawn.
"""

from __future__ import annotations

import argparse
import os
import sys
import uvicorn
import torch_xla.distributed.xla_multiprocessing as xmp

from frontiersplit.worker import create_worker_app


def _stage_worker_entry(index: int, total_stages: int, model_name: str, use_binary: bool, quantize_activations: bool):
    """Entry point for each spawned TPU core worker process."""
    stage_id = index
    is_final = (stage_id == total_stages - 1)
    port = 50051 + stage_id
    tcp_port = (50151 + stage_id) if use_binary else None
    downstream_port = 50051 + stage_id + 1
    downstream_tcp_port = 50151 + stage_id + 1

    downstream_url = None if is_final else f"http://127.0.0.1:{downstream_port}"
    downstream_tcp = None if (is_final or not use_binary) else f"127.0.0.1:{downstream_tcp_port}"

    print(f"[TPU Stage {stage_id}/{total_stages}] Initializing on TPU core (HTTP :{port}, TCP :{tcp_port})...")
    app = create_worker_app(
        stage_id=stage_id,
        total_stages=total_stages,
        downstream_url=downstream_url,
        downstream_tcp=downstream_tcp,
        model_name_or_path=model_name,
        device="xla",
        tcp_port=tcp_port,
        quantize_activations=quantize_activations,
    )
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def main():
    parser = argparse.ArgumentParser(description="FrontierSplit TPU Multi-Core Runner")
    parser.add_argument("--total-stages", type=int, default=8, help="Total pipeline stages (default: 8)")
    parser.add_argument("--model-name", type=str, default="mistralai/Mixtral-8x7B-v0.1", help="Model ID")
    parser.add_argument("--use-binary", action="store_true", default=True, help="Enable persistent binary TCP transport")
    parser.add_argument("--disable-binary", dest="use_binary", action="store_false", help="Disable persistent binary TCP transport")
    parser.add_argument("--quantize-activations", action="store_true", default=False, help="Enable dynamic INT8 activation quantization")
    args = parser.parse_args()

    print("==================================================================")
    print(" FrontierSplit Cloud TPU Multi-Core Runner")
    print(f" Accelerator: v5litepod-8 (8 TPU Cores via xmp.spawn)")
    print(f" Model:       {args.model_name}")
    print(f" Binary TCP:  {'ENABLED' if args.use_binary else 'DISABLED'}")
    print(f" Quantize:    {'ENABLED (INT8)' if args.quantize_activations else 'DISABLED (FP16)'}")
    print("==================================================================")

    xmp.spawn(
        _stage_worker_entry,
        args=(args.total_stages, args.model_name, args.use_binary, args.quantize_activations),
    )


if __name__ == "__main__":
    main()
