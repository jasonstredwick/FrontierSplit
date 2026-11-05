import torch

from frontiersplit.models import interface as fs_model_interface
from frontiersplit.runners import interface as fs_runner_interface


class LocalRunner(fs_runner_interface.Runner):
    def __init__(self, *, model: fs_model_interface.Model, output_device: torch.device):
        self._model = model
        self._output_device = output_device

    def Execute(self, *, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Tensor.to returns the existing tensor when its device already matches.
        device_inputs = {name: tensor.to(device=self._model.model_spec.device) for name, tensor in inputs.items()}
        outputs = self._model.Execute(inputs=device_inputs)
        return {name: tensor.to(device=self._output_device) for name, tensor in outputs.items()}
