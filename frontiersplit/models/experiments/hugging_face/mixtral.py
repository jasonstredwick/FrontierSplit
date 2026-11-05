import torch

from transformers.models.mixtral import modeling_mixtral as _modeling_mixtral

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
        model_class=_modeling_mixtral.MixtralForCausalLM,
        checkpoint=checkpoint,
        revision=revision,
        dtype=dtype,
        device=device,
        attention=attention,
    )
