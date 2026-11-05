import abc
import torch


class Runner(abc.ABC):
    @abc.abstractmethod
    def Execute(self, *, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError
