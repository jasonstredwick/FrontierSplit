import abc
import dataclasses
import torch


@dataclasses.dataclass(frozen=True, kw_only=True)
class TensorSpec:
    dtype: torch.dtype
    shape: tuple[int | str, ...]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModelSpec:
    device: torch.device
    inputs: dict[str, TensorSpec]
    outputs: dict[str, TensorSpec]


class Model(abc.ABC):
    def __init__(self, *, model_spec: ModelSpec):
        self.model_spec = model_spec

    @abc.abstractmethod
    def Execute(self, *, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError


def ValidateInputs(*, model_spec: ModelSpec, inputs: dict[str, torch.Tensor]) -> None:
    if inputs.keys() != model_spec.inputs.keys():
        raise ValueError(f"Expected input tensors {sorted(model_spec.inputs)}, got {sorted(inputs)}")
    dimensions: dict[str, int] = {}
    for name, spec in model_spec.inputs.items():
        tensor = inputs[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device != model_spec.device:
            raise ValueError(f"{name} must reside on {model_spec.device}, got {tensor.device}")
        if tensor.dtype != spec.dtype:
            raise ValueError(f"{name} must have dtype {spec.dtype}, got {tensor.dtype}")
        if tensor.ndim != len(spec.shape):
            raise ValueError(f"{name} must have shape {spec.shape}, got {tuple(tensor.shape)}")
        for expected, size in zip(spec.shape, tensor.shape, strict=True):
            required = dimensions.setdefault(expected, size) if isinstance(expected, str) else expected
            if size != required:
                raise ValueError(f"{name} must have shape {spec.shape}, got {tuple(tensor.shape)}")
