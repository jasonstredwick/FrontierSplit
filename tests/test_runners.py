import pytest
import torch

from frontiersplit.models import interface as fs_model_interface
from frontiersplit.runners import local as fs_local


class ArithmeticModel(fs_model_interface.Model):
    def __init__(self, *, device: torch.device):
        spec = fs_model_interface.TensorSpec(dtype=torch.float32, shape=(2, 3))
        super().__init__(
            device=device, input_specs={"a": spec, "b": spec}, output_specs={"a": spec, "b": spec},
        )

    def Execute(self, *, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self._ValidateInputs(inputs=inputs)
        return {"a": inputs["a"], "b": inputs["a"] + inputs["b"]}


def test_local_runners_connect_without_copying_on_the_same_device():
    device = torch.device("cpu")
    first = fs_local.LocalRunner(model=ArithmeticModel(device=device), output_device=device)
    second = fs_local.LocalRunner(model=ArithmeticModel(device=device), output_device=device)
    a = torch.ones((2, 3))
    b = torch.full((2, 3), 2.0)
    intermediate = first.Execute(inputs={"a": a, "b": b})
    output = second.Execute(inputs=intermediate)
    assert intermediate["a"] is a
    assert output["a"] is a
    torch.testing.assert_close(output["b"], torch.full((2, 3), 4.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device")
def test_local_runners_transfer_only_at_explicit_device_boundaries():
    device = torch.device("cuda:0")
    model = ArithmeticModel(device=device)
    first = fs_local.LocalRunner(model=model, output_device=device)
    second = fs_local.LocalRunner(model=model, output_device=torch.device("cpu"))
    inputs = {"a": torch.ones((2, 3)), "b": torch.full((2, 3), 2.0)}
    with pytest.raises(ValueError, match="must reside"):
        model.Execute(inputs=inputs)
    intermediate = first.Execute(inputs=inputs)
    assert all(tensor.device == device for tensor in intermediate.values())
    assert model.Execute(inputs=intermediate)["a"] is intermediate["a"]
    output = second.Execute(inputs=intermediate)
    assert all(tensor.device == torch.device("cpu") for tensor in output.values())
    torch.testing.assert_close(output["b"], torch.full((2, 3), 4.0))
