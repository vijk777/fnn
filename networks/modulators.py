import torch
from torch import nn
from typing import Sequence

from .containers import Module
from .elements import Linear


class Modulator(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)


class LSTM(Modulator):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
        )

        linear = (
            lambda: Linear(out_features=self.out_features)
            .add(in_features=self.in_features)
            .add(in_features=self.out_features)
        )
        self.proj_i = linear()
        self.proj_f = linear()
        self.proj_g = linear()
        self.proj_o = linear()

        self.gain = nn.Parameter(torch.zeros(self.out_features))

        self._past = dict()

    def _reset(self):
        self._past.clear()

    def forward(self, x: torch.Tensor):
        """
        Args:
            inputs (Sequence of torch.Tensors): shape = [n, f]
        Returns:
            (torch.Tensor): shape = [n, f']
        """
        if self._past:
            h = self._past["h"]
            c = self._past["c"]
        else:
            h = c = torch.zeros(1, self.out_features, device=self.device)

        i = torch.sigmoid(self.proj_i([x, h]))
        f = torch.sigmoid(self.proj_f([x, h]))
        g = torch.tanh(self.proj_g([x, h]))
        o = torch.sigmoid(self.proj_o([x, h]))

        c = f * c + i * g
        h = o * torch.tanh(c)

        self._past["c"] = c
        self._past["h"] = h

        return h * self.gain
