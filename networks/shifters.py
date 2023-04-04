import torch
from torch import nn
from typing import Sequence, Optional

from .containers import Module, ModuleList
from .elements import Linear, nonlinearity
from .standardization import EyePosition


def default_shifter(
    eye_position: EyePosition,
):
    return MLP(
        eye_position=eye_position,
        out_features=[8, 8],
        nonlinear="gelu",
    )


class Shifter(Module):
    def __init__(
        self,
        eye_position: EyePosition,
        out_features: int,
    ):
        super().__init__()
        self.eye_position = eye_position
        self.out_features = int(out_features)


class MLP(Shifter):
    def __init__(
        self,
        eye_position: EyePosition,
        out_features: Sequence[int],
        nonlinear: Optional[str] = None,
    ):
        super().__init__(
            eye_position=eye_position,
            out_features=out_features[-1],
        )

        self.layers = ModuleList()

        in_features = self.eye_position.n_features
        for features in out_features:

            linear = Linear(out_features=features).add(in_features=in_features)
            in_features = features

            self.layers.append(linear)

        self.nonlinear, self.gamma = nonlinearity(nonlinear)

    def forward(self, eye_position: Optional[torch.Tensor] = None):
        """
        Args:
            eye_position (torch.Tensor): shape = [n, f]
        Returns:
            (torch.Tensor): shape = [n, f']
        """
        x = self.eye_position(eye_position)

        for layer in self.layers:
            x = layer([x])
            x = self.nonlinear(x).mul(self.gamma)

        return x
