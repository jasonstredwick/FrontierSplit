import torch

from transformers.models.mistral import modeling_mistral as _modeling_mistral

from frontiersplit.models import interface as fs_interface
from frontiersplit.models.experiments.hugging_face import _transformers as fs_transformers


def LoadModel(
    *,
    checkpoint: str,
    revision: str,
    dtype: torch.dtype,
    device: torch.device,
    attention: str
) -> fs_interface.Model:
    return fs_transformers.LoadModel(
        model_class=_modeling_mistral.MistralForCausalLM,
        checkpoint=checkpoint,
        revision=revision,
        dtype=dtype,
        device=device,
        attention=attention,
    )
