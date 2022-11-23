from abc import ABC
from typing import Tuple

import torch
from torch import Tensor
from torchmetrics import Metric


class Min(Metric, ABC):
    full_state_update: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("min", torch.tensor(float("inf")), persistent=True)  # pylint: disable=not-callable
        self.min = torch.tensor(float("inf"))  # pylint: disable=not-callable

    def update(self, a) -> None:
        print("cccccccc")
        self.min = torch.min(a)

    def compute(self) -> Tuple[Tensor, Tensor]:
        """Return min and max values."""
        return self.min
