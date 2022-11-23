from abc import ABC
from typing import Tuple

import torch
from torch import Tensor
from torchmetrics import Metric


class MinMax(Metric, ABC):
    full_state_update: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("min", torch.tensor(float("inf")), persistent=True)  # pylint: disable=not-callable
        self.add_state("max", torch.tensor(float("-inf")), persistent=True)  # pylint: disable=not-callable

        self.min = torch.tensor(float("inf"))  # pylint: disable=not-callable
        self.max = torch.tensor(float("-inf"))  # pylint: disable=not-callable

    def update(self, a, b) -> None:
        self.min = torch.min(a)
        self.max = torch.min(b)

    def compute(self) -> Tuple[Tensor, Tensor]:
        """Return min and max values."""
        return self.min, self.max
