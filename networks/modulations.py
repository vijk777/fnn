import torch
from typing import Optional

from .containers import Module
from .variables import Behavior
from .elements import Linear


class Modulation(Module):
    def __init__(self, behavior: Behavior):
        super().__init__()
        self.behavior = behavior

    @property
    def out_channels(self):
        raise NotImplementedError

    def forward(self, behavior: Optional[torch.Tensor] = None):
        """
        Args:
            behavior (torch.Tensor) : shape = [n, f]
        Returns:
            (torch.Tensor)          : shape = [n, c, 1, 1]
        """
        raise NotImplementedError()


class LSTM(Modulation):
    def __init__(self, behavior: Behavior, size: int):
        super().__init__(behavior=behavior)

        linear = (
            lambda: Linear(out_features=self.out_channels)
            .add(in_features=self.behavior.n_features)
            .add(in_features=self.out_channels)
        )
        self.proj_i = linear()
        self.proj_f = linear()
        self.proj_g = linear()
        self.proj_o = linear()

        self._past = dict()

    def _reset(self):
        self._past.clear()

    @property
    def out_channels(self):
        return self.size

    def forward(self, behavior: Optional[torch.Tensor] = None):
        """
        Args:
            behavior (torch.Tensor) : shape = [n, f]
        Returns:
            (torch.Tensor)          : shape = [n, c, 1, 1]
        """
        if self._past:
            h = self._past["h"]
            c = self._past["c"]
        else:
            h = c = torch.zeros(1, self.out_channels, device=self.device)

        x = self.behavior(behavior)

        i = torch.sigmoid(self.proj_i([x, h]))
        f = torch.sigmoid(self.proj_f([x, h]))
        g = torch.tanh(self.proj_g([x, h]))
        o = torch.sigmoid(self.proj_o([x, h]))

        c = f * c + i * g
        h = o * torch.tanh(c)

        self._past["c"] = c
        self._past["h"] = h

        return h[:, :, None, None]
