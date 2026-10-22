"""FrontierSplit Model Specifications and Architectural Profiles.

Maintains model-specific architectural configurations (context windows, generation ceilings,
stop sequences, layer counts, MoE routing properties) so every model operates strictly
under its native standards without hardcoded assumptions or artificial limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class ModelSpec:
    """Architectural and operational specification for an LLM."""
    model_id: str
    architecture: str                      # "dense_transformer" or "moe_transformer"
    context_window: int                   # Physical context capacity (e.g. 32768, 131072)
    max_generation_tokens: int            # Standard output generation ceiling (e.g. 4096, 8192)
    stop_tokens: List[str]                # Model-specific stop tokens
    stop_token_ids: Set[int] = field(default_factory=set)
    num_layers: int = 32
    hidden_size: int = 4096
    num_experts: Optional[int] = None      # Total experts (MoE)
    top_k: Optional[int] = None            # Active experts per token (MoE)
    default_temperature: float = 0.7


# Known Model Architecture Registry
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # Mistral 7B Series
    "mistralai/Mistral-7B-Instruct-v0.3": ModelSpec(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        architecture="dense_transformer",
        context_window=32768,
        max_generation_tokens=4096,
        stop_tokens=["</s>", "[INST]"],
        num_layers=32,
        hidden_size=4096,
    ),
    "mistralai/Mistral-7B-Instruct-v0.2": ModelSpec(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        architecture="dense_transformer",
        context_window=32768,
        max_generation_tokens=4096,
        stop_tokens=["</s>"],
        num_layers=32,
        hidden_size=4096,
    ),
    # Mixtral 8x7B MoE Series
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ModelSpec(
        model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        architecture="moe_transformer",
        context_window=32768,
        max_generation_tokens=4096,
        stop_tokens=["</s>"],
        num_layers=32,
        hidden_size=4096,
        num_experts=8,
        top_k=2,
    ),
    # Llama 3.1 Series
    "meta-llama/Llama-3.1-8B-Instruct": ModelSpec(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        architecture="dense_transformer",
        context_window=131072,
        max_generation_tokens=8192,
        stop_tokens=["<|eot_id|>", "<|end_of_text|>"],
        num_layers=32,
        hidden_size=4096,
    ),
    "meta-llama/Llama-3.1-70B-Instruct": ModelSpec(
        model_id="meta-llama/Llama-3.1-70B-Instruct",
        architecture="dense_transformer",
        context_window=131072,
        max_generation_tokens=8192,
        stop_tokens=["<|eot_id|>", "<|end_of_text|>"],
        num_layers=80,
        hidden_size=8192,
    ),
    # Qwen 2.5 Series
    "Qwen/Qwen2.5-7B-Instruct": ModelSpec(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        architecture="dense_transformer",
        context_window=131072,
        max_generation_tokens=8192,
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
        num_layers=28,
        hidden_size=3584,
    ),
    "Qwen/Qwen2.5-Coder-7B-Instruct": ModelSpec(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        architecture="dense_transformer",
        context_window=131072,
        max_generation_tokens=8192,
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
        num_layers=28,
        hidden_size=3584,
    ),
}


def resolve_model_spec(model_id: str, tokenizer: Optional[Any] = None) -> ModelSpec:
    """Resolve ModelSpec using registry with dynamic AutoConfig fallback for any model."""
    spec = MODEL_REGISTRY.get(model_id)

    if spec is None:
        # Fallback for unlisted or custom models: attempt dynamic AutoConfig introspection
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id)
            context = getattr(config, "max_position_embeddings", 32768)
            num_layers = getattr(config, "num_hidden_layers", 32)
            hidden = getattr(config, "hidden_size", 4096)
            num_exp = getattr(config, "num_local_experts", getattr(config, "num_experts", None))
            top_k = getattr(config, "num_experts_per_tok", getattr(config, "top_k", None))
            arch = "moe_transformer" if num_exp else "dense_transformer"
            gen_ceiling = min(8192, context // 4)

            spec = ModelSpec(
                model_id=model_id,
                architecture=arch,
                context_window=context,
                max_generation_tokens=max(2048, gen_ceiling),
                stop_tokens=["</s>", "<|endoftext|>"],
                num_layers=num_layers,
                hidden_size=hidden,
                num_experts=num_exp,
                top_k=top_k,
            )
        except Exception:
            # Generic safe default if offline or transformers not installed
            spec = ModelSpec(
                model_id=model_id,
                architecture="dense_transformer",
                context_window=32768,
                max_generation_tokens=4096,
                stop_tokens=["</s>"],
                num_layers=32,
                hidden_size=4096,
            )

    # If tokenizer is supplied, dynamically populate stop_token_ids
    if tokenizer is not None:
        ids = set(spec.stop_token_ids)
        raw_eos = getattr(tokenizer, "eos_token_id", None)
        if isinstance(raw_eos, (list, tuple, set)):
            ids.update(raw_eos)
        elif raw_eos is not None:
            ids.add(raw_eos)

        for st in spec.stop_tokens:
            try:
                tid = tokenizer.convert_tokens_to_ids(st)
                if tid is not None and tid != getattr(tokenizer, "unk_token_id", None):
                    ids.add(tid)
            except Exception:
                pass
        spec.stop_token_ids = ids

    return spec
