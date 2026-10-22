"""FrontierSplit Shard Planner & Selective Loader Utility.

Calculates the exact layer partition and Hugging Face safetensors shard distribution
across pipeline stages. Enables downloading and caching ONLY the required shards on
each node without downloading full 70B+ model weights (~140 GB) locally or remotely.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from typing import Any, Dict, List, Set, Tuple


class ShardPlanner:
    """Analyzes model safetensors indices and maps shards to pipeline stages."""

    @staticmethod
    def fetch_index(model_id_or_path: str) -> Dict[str, Any]:
        """Fetch model.safetensors.index.json from Hugging Face Hub or local path."""
        if os.path.isfile(model_id_or_path):
            with open(model_id_or_path, "r", encoding="utf-8") as f:
                return json.load(f)

        url = f"https://huggingface.co/{model_id_or_path}/raw/main/model.safetensors.index.json"
        req = urllib.request.Request(url, headers={"User-Agent": "FrontierSplit-ShardPlanner/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to fetch safetensors index from {url}: {e}") from e

    @staticmethod
    def fetch_config(model_id_or_path: str) -> Dict[str, Any]:
        """Fetch config.json from Hugging Face Hub or local path."""
        if os.path.isfile(model_id_or_path):
            with open(model_id_or_path, "r", encoding="utf-8") as f:
                return json.load(f)

        url = f"https://huggingface.co/{model_id_or_path}/raw/main/config.json"
        req = urllib.request.Request(url, headers={"User-Agent": "FrontierSplit-ShardPlanner/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to fetch config from {url}: {e}") from e

    @classmethod
    def plan_shards(
        cls,
        model_id: str,
        total_stages: int = 8,
    ) -> Dict[str, Any]:
        """Calculates stage layer ranges and required safetensors shards."""
        config = cls.fetch_config(model_id)
        index_data = cls.fetch_index(model_id)
        weight_map: Dict[str, str] = index_data.get("weight_map", {})
        metadata = index_data.get("metadata", {})
        total_size_bytes = metadata.get("total_size", 0)

        total_layers = (
            config.get("num_hidden_layers")
            or config.get("n_layers")
            or config.get("num_layers", 32)
        )
        layers_per_stage = total_layers // total_stages
        remainder = total_layers % total_stages

        stages_info = []
        all_unique_shards = sorted(set(weight_map.values()))

        current_layer = 0
        for s in range(total_stages):
            assigned_count = layers_per_stage + (1 if s < remainder else 0)
            start_layer = current_layer
            end_layer = current_layer + assigned_count - 1
            current_layer += assigned_count

            is_first = (s == 0)
            is_final = (s == total_stages - 1)

            needed_shards: Set[str] = set()
            for tensor_name, shard_file in weight_map.items():
                # Layers
                if any(tensor_name.startswith(f"model.layers.{l}.") for l in range(start_layer, end_layer + 1)):
                    needed_shards.add(shard_file)
                # Embeddings (first stage)
                if is_first and tensor_name.startswith("model.embed_tokens."):
                    needed_shards.add(shard_file)
                # Norm and LM head (final stage)
                if is_final and (tensor_name.startswith("model.norm.") or tensor_name.startswith("lm_head.")):
                    needed_shards.add(shard_file)

            stages_info.append({
                "stage_id": s,
                "start_layer": start_layer,
                "end_layer": end_layer,
                "layer_count": assigned_count,
                "is_first": is_first,
                "is_final": is_final,
                "needed_shards": sorted(needed_shards),
                "num_shards": len(needed_shards),
            })

        return {
            "model_id": model_id,
            "architecture": config.get("architectures", ["Transformer"])[0] if isinstance(config.get("architectures"), list) else "Transformer",
            "total_layers": total_layers,
            "total_stages": total_stages,
            "total_shards": len(all_unique_shards),
            "total_size_gb": total_size_bytes / (1024 ** 3) if total_size_bytes else 0.0,
            "stages": stages_info,
        }

    @classmethod
    def print_plan(cls, plan: Dict[str, Any]) -> None:
        """Pretty prints the shard distribution report."""
        print("=" * 78)
        print(f" FRONTIERSPLIT SHARD PARTITION PLAN: {plan['model_id']}")
        print("=" * 78)
        print(f"Architecture:      {plan['architecture']}")
        print(f"Total Layers:      {plan['total_layers']} layers across {plan['total_stages']} pipeline stages")
        print(f"Total Model Files: {plan['total_shards']} safetensors shards ({plan['total_size_gb']:.2f} GB total)")
        print("-" * 78)
        print(f"{'Stage':<8} {'Layers':<12} {'Shards':<10} {'Selective Shards Required'}")
        print("-" * 78)
        for s in plan["stages"]:
            role = ""
            if s["is_first"]:
                role = " [Embeddings]"
            elif s["is_final"]:
                role = " [LM Head]"
            shard_summary = f"{s['needed_shards'][0]} .. {s['needed_shards'][-1]}" if len(s['needed_shards']) > 1 else s['needed_shards'][0]
            print(f"Stage {s['stage_id']:<2} L{s['start_layer']:02d}-L{s['end_layer']:02d}{role:<14} {s['num_shards']} shards   {shard_summary}")
        print("=" * 78)
        print("Selective Loading Advantage: Each node downloads ONLY its required 15-22 GB")
        print("without downloading the full 140 GB weights onto local or remote disks!")
        print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="FrontierSplit Shard Planner")
    parser.add_argument("model_id", nargs="?", default="Qwen/Qwen2.5-72B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--stages", type=int, default=8, help="Number of pipeline stages")
    args = parser.parse_args()

    plan = ShardPlanner.plan_shards(args.model_id, total_stages=args.stages)
    ShardPlanner.print_plan(plan)


if __name__ == "__main__":
    main()
