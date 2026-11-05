import pathlib
import torch
import transformers

from frontiersplit.models import interface as fs_interface


_TRANSFORMERS_VERSION = "5.20.2"


class _TransformersModel(fs_interface.Model):
    def __init__(self, *, implementation: transformers.PreTrainedModel):
        input_spec = fs_interface.TensorSpec(dtype=torch.int64, shape=("batch", "sequence"))
        output_spec = fs_interface.TensorSpec(
            dtype=implementation.dtype,
            shape=("batch", "sequence", implementation.config.vocab_size),
        )

        self._implementation = implementation
        super().__init__(
            model_spec=fs_interface.ModelSpec(
                device=implementation.device,
                inputs={"attention_mask": input_spec, "input_ids": input_spec, "position_ids": input_spec},
                outputs={"logits": output_spec}
            )
        )

    def Execute(self, *, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch, sequence = inputs["input_ids"].shape
        if batch < 1 or not 1 <= sequence <= self._implementation.config.max_position_embeddings:
            raise ValueError("Expected a nonempty batch and sequence within the model's context limit")
        with torch.inference_mode():
            outputs = self._implementation(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                position_ids=inputs["position_ids"],
                use_cache=False,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
        return {"logits": outputs.logits}


def LoadModel(
    *,
    model_class: type[transformers.PreTrainedModel],
    checkpoint: str,
    revision: str,
    dtype: torch.dtype,
    device: torch.device,
    attention: str
) -> fs_interface.Model:
    if transformers.__version__ != _TRANSFORMERS_VERSION:
        raise ValueError(f"These experiments require transformers=={_TRANSFORMERS_VERSION}")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Reference experiments require BF16 or FP16 weights")
    if device.type not in ("cuda"):
        raise ValueError("These experiments support a single CUDA device")
    cache = pathlib.Path.cwd() / ".__pycache__" / "huggingface" / "hub"
    config = transformers.AutoConfig.from_pretrained(
        checkpoint, revision=revision, cache_dir=cache, trust_remote_code=False,
    )
    if not isinstance(config, model_class.config_class):
        raise ValueError(f"Checkpoint architecture {config.model_type} does not match {model_class.__name__}")
    implementation, loading = model_class.from_pretrained(
        checkpoint,
        config=config,
        revision=revision,
        cache_dir=cache,
        trust_remote_code=False,
        dtype=dtype,
        device_map={"": device},
        attn_implementation=attention,
        output_loading_info=True,
    )
    if loading["missing_keys"] or loading["unexpected_keys"] or loading["mismatched_keys"] or loading["error_msgs"]:
        raise ValueError(f"Checkpoint did not load completely: {loading}")
    return _TransformersModel(implementation=implementation.eval())
