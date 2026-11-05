import enum
import torch

from frontiersplit.models import interface as fs_interface
from frontiersplit.models.experiments.baseline import mistral as fs_baseline_mistral
from frontiersplit.models.experiments.baseline import mixtral as fs_baseline_mixtral
from frontiersplit.models.experiments.hugging_face import mistral as fs_hugging_face_mistral
from frontiersplit.models.experiments.hugging_face import mixtral as fs_hugging_face_mixtral


class ExperimentName(enum.StrEnum):
    BASELINE = "baseline"
    HUGGING_FACE = "hugging_face"


class ModelName(enum.StrEnum):
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"


_MODEL_EXPERIMENTS = {
    ModelName.MISTRAL: {
        ExperimentName.BASELINE: fs_baseline_mistral.LoadModel,
        ExperimentName.HUGGING_FACE: fs_hugging_face_mistral.LoadModel,
    },
    ModelName.MIXTRAL: {
        ExperimentName.BASELINE: fs_baseline_mixtral.LoadModel,
        ExperimentName.HUGGING_FACE: fs_hugging_face_mixtral.LoadModel,
    },
}


def GetModel() -> fs_interface.Model:
    raise NotImplementedError("No primary model implementation has been selected")


def GetModelExperiment(
    *,
    model_name: ModelName,
    experiment_name: ExperimentName,
    checkpoint: str,
    revision: str,
    dtype: torch.dtype,
    device: torch.device,
    attention: str
) -> fs_interface.Model:
    try:
        load_model = _MODEL_EXPERIMENTS[model_name][experiment_name]
    except KeyError as error:
        raise ValueError(f"Unknown model experiment: {model_name}/{experiment_name}") from error
    return load_model(
        checkpoint=checkpoint,
        revision=revision,
        dtype=dtype,
        device=device,
        attention=attention,
    )
