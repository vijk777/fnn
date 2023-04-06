import torch

from .containers import Module
from .elements import Linear


class Modulation(Module):
    def __init__(
        self,
        behavior,
    ):
        """
        Parameters
        ----------
        behavior : .variables.Behavior
            behavior variable
        """
        super().__init__()
        self.behavior = behavior

    @property
    def out_features(self):
        raise NotImplementedError

    def forward(self, behavior=None):
        """
        Parameters
        ----------
        behavior : Tensor | None
            shape = [n, f]

        Returns
        -------
        Tensor
            shape = [n, f']
        """
        raise NotImplementedError()


class LSTM(Modulation):
    def __init__(
        self,
        behavior,
        size,
    ):
        """
        Parameters
        ----------
        behavior : .variables.Behavior
            behavior variable
        size : int
            size of LSTM
        """
        super().__init__(behavior=behavior)

        self.size = int(size)

        features = self.behavior.features
        linear = lambda: Linear(out_features=self.size).add(in_features=features).add(in_features=self.size)

        self.proj_i = linear()
        self.proj_f = linear()
        self.proj_g = linear()
        self.proj_o = linear()

        self._past = dict()

    def _reset(self):
        self._past.clear()

    @property
    def out_features(self):
        return self.size

    def forward(self, behavior=None):
        """
        Parameters
        ----------
        behavior : Tensor | None
            shape = [n, f]

        Returns
        -------
        Tensor
            shape = [n, f']
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

        return h
