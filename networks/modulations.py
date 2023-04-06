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
    def features(self):
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
        features,
    ):
        """
        Parameters
        ----------
        behavior : .variables.Behavior
            behavior variable
        features : int
            LSTM features
        """
        super().__init__(behavior=behavior)

        self._features = int(features)

        features = self.behavior.features
        linear = lambda: Linear(out_features=self.features).add(in_features=features).add(in_features=self.features)

        self.proj_i = linear()
        self.proj_f = linear()
        self.proj_g = linear()
        self.proj_o = linear()

        self._past = dict()

    def _reset(self):
        self._past.clear()

    @property
    def features(self):
        return self._features

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
            h = c = torch.zeros(1, self.features, device=self.device)

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
