import torch
from torch import nn
from typing import Sequence, Optional

from .containers import Module, ModuleList
from .elements import Linear, nonlinearity


class Shifter(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)


class MLP(Shifter):
    def __init__(
        self,
        in_features: int,
        out_features: Sequence[int],
        nonlinear: Optional[str] = None,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features[-1],
        )

        self.layers = ModuleList()

        in_features = self.in_features
        for out_features in out_features:

            linear = Linear(out_features=out_features).add(in_features=in_features)
            in_features = out_features

            self.layers.append(linear)

        self.nonlinear, self.gamma = nonlinearity(nonlinear)

        self.gain = nn.Parameter(torch.zeros(self.out_features))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): shape = [n, f]
        Returns:
            (torch.Tensor): shape = [n, f']
        """
        for layer in self.layers:
            x = layer([x])
            x = self.nonlinear(x).mul(self.gamma)

        return x.mul(self.gain)
