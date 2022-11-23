from abc import ABC

import torch
import torchmetrics


class MM(torchmetrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("kkk", torch.tensor(float("inf")), persistent=True)
        self.add_state("lll", torch.tensor(float("-inf")), persistent=True)

        self.kkk = torch.tensor(float("inf"))
        self.lll = torch.tensor(float("-inf"))

    def update(self, a, b) -> None:
        print("cccc")
        self.kkk = torch.min(a)
        self.lll = torch.min(b)

    def compute(self):
        print("dddd")
        return self.kkk, self.lll
