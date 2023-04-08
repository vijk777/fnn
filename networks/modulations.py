import torch

from .containers import Module
from .elements import Linear


class Modulation(Module):
    def init(self, behaviors):
        """
        Parameters
        ----------
        behaviors : int
            behavior features
        """
        raise NotImplementedError()

    @property
    def features(self):
        raise NotImplementedError()

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
    def __init__(self, features):
        """
        Parameters
        ----------
        features : int
            LSTM features
        """
        super().__init__()

        self._features = int(features)

        linear = lambda: Linear(out_features=self.features).add(in_features=self.features)
        self.proj_i = linear()
        self.proj_f = linear()
        self.proj_g = linear()
        self.proj_o = linear()

        self._past = dict()

    def _reset(self):
        self._past.clear()

    def init(self, behaviors):
        """
        Parameters
        ----------
        behaviors : int
            behavior features
        """
        self.proj_i.add_input(in_features=behaviors)
        self.proj_f.add_input(in_features=behaviors)
        self.proj_g.add_input(in_features=behaviors)
        self.proj_o.add_input(in_features=behaviors)

    @property
    def features(self):
        return self._features

    def forward(self, behavior):
        """
        Parameters
        ----------
        behavior : Tensor
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
            h = c = torch.zeros(behavior.size(0), self.features, device=self.device)

        inputs = [h, behavior]

        i = torch.sigmoid(self.proj_i(inputs))
        f = torch.sigmoid(self.proj_f(inputs))
        g = torch.tanh(self.proj_g(inputs))
        o = torch.sigmoid(self.proj_o(inputs))

        c = f * c + i * g
        h = o * torch.tanh(c)

        self._past["c"] = c
        self._past["h"] = h

        return h
