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

    @property
    def streams(self):
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
    def __init__(self, features, streams):
        """
        Parameters
        ----------
        features : int
            number of features
        streams : int
            number of streams
        """
        super().__init__()

        self.proj_i = Linear(features=features, streams=streams).add_input(features=features)
        self.proj_f = Linear(features=features, streams=streams).add_input(features=features)
        self.proj_g = Linear(features=features, streams=streams).add_input(features=features)
        self.proj_o = Linear(features=features, streams=streams).add_input(features=features)

        self._features = int(features)
        self._streams = int(streams)

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
        self.proj_i.add_input(features=behaviors)
        self.proj_f.add_input(features=behaviors)
        self.proj_g.add_input(features=behaviors)
        self.proj_o.add_input(features=behaviors)

    @property
    def features(self):
        return self._features

    @property
    def streams(self):
        return self._streams

    def forward(self, behavior, stream=None):
        """
        Parameters
        ----------
        behavior : Tensor
            shape = [n, f] -- stream is None
                or
            shape = [n, f // s] -- stream is not None
        stream : int | None
            specific stream index (int) or all streams (None)

        Returns
        -------
        Tensor
            shape = [n, f'] -- stream is None
                or
            shape = [n, f' // s] -- stream is not None
        """
        if self._past:
            assert self._past["stream"] == stream

            h = self._past["h"]
            c = self._past["c"]
        else:
            self._past["stream"] = stream

            features = self.features if stream is None else self.features // self.streams
            h = c = torch.zeros(behavior.size(0), features, device=self.device)

        inputs = [h, behavior]

        i = torch.sigmoid(self.proj_i(inputs, stream=stream))
        f = torch.sigmoid(self.proj_f(inputs, stream=stream))
        g = torch.tanh(self.proj_g(inputs, stream=stream))
        o = torch.sigmoid(self.proj_o(inputs, stream=stream))

        c = f * c + i * g
        h = o * torch.tanh(c)

        self._past["c"] = c
        self._past["h"] = h
        self._past["stream"] = stream

        return h
