import torch

from .containers import Module
from .elements import Linear


class Modulation(Module):
    @property
    def features(self):
        """
        Returns
        -------
        int
            output features, f'
        """
        raise NotImplementedError()

    def init(self, behaviors, streams):
        """
        Parameters
        ----------
        behaviors : int
            behavior features, f
        streams : int
            number of streams, s
        """
        raise NotImplementedError()

    def forward(self, behavior, stream=None):
        """
        Parameters
        ----------
        behavior : Tensor
            shape = [n, f * s] -- stream is None
                or
            shape = [n, f] -- stream is int
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            shape = [n, f' * s] -- stream is None
                or
            shape = [n, f'] -- stream is int
        """
        raise NotImplementedError()


class LSTM(Modulation):
    def __init__(self, features):
        """
        Parameters
        ----------
        features : int
            output features, f'
        """
        super().__init__()

        self._features = int(features)
        self._past = dict()

    def _reset(self):
        self._past.clear()

    @property
    def features(self):
        """
        Returns
        -------
        int
            output features, f'
        """
        return self._features

    def init(self, behaviors, streams):
        """
        Parameters
        ----------
        behaviors : int
            behavior features, f
        streams : int
            number of streams, s
        """
        self.proj_i = (
            Linear(features=self.features, streams=streams)
            .add_input(features=behaviors)
            .add_input(features=self.features)
        )
        self.proj_f = (
            Linear(features=self.features, streams=streams)
            .add_input(features=behaviors)
            .add_input(features=self.features)
        )
        self.proj_g = (
            Linear(features=self.features, streams=streams)
            .add_input(features=behaviors)
            .add_input(features=self.features)
        )
        self.proj_o = (
            Linear(features=self.features, streams=streams)
            .add_input(features=behaviors)
            .add_input(features=self.features)
        )
        self.streams = int(streams)

    def forward(self, behavior, stream=None):
        """
        Parameters
        ----------
        behavior : Tensor
            shape = [n, f]
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            shape = [n, f' * s] -- stream is None
                or
            shape = [n, f'] -- stream is int
        """
        if self._past:
            assert self._past["stream"] == stream

            h = self._past["h"]
            c = self._past["c"]
        else:
            self._past["stream"] = stream

            features = self.features * self.streams if stream is None else self.features
            h = c = torch.zeros(behavior.size(0), features, device=self.device)

        if stream is None:
            behavior = behavior.repeat(1, self.streams)

        inputs = [behavior, h]

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
