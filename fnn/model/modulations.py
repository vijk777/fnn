import torch
from .modules import Module
from .elements import Linear


class Modulation(Module):
    @property
    def features(self):
        """
        Returns
        -------
        int
            modulation features (M)
        """
        raise NotImplementedError()

    def _init(self, modulations, streams):
        """
        Parameters
        ----------
        modulations : int
            modulation inputs (I)
        streams : int
            number of streams (S)
        """
        raise NotImplementedError()

    def forward(self, modulation, stream=None):
        """
        Parameters
        ----------
        modulation : Tensor
            [N, S*I] -- stream is None
                or
            [N, I] -- stream is int
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            [N, S*M] -- stream is None
                or
            [N, M] -- stream is int
        """
        raise NotImplementedError()


class Lstm(Modulation):
    def __init__(self, features):
        """
        Parameters
        ----------
        features : int
            lstm features
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
            modulation features (M)
        """
        return self._features

    def _init(self, modulations, streams):
        """
        Parameters
        ----------
        modulations : int
            number of inputs (I)
        streams : int
            number of streams (S)
        """
        self.modulations = int(modulations)
        self.streams = int(streams)
        self.proj_i = (
            Linear(features=self.features, streams=streams)
            .add_input(features=self.modulations)
            .add_input(features=self.features)
        )
        self.proj_f = (
            Linear(features=self.features, streams=streams)
            .add_input(features=self.modulations)
            .add_input(features=self.features)
        )
        self.proj_g = (
            Linear(features=self.features, streams=streams)
            .add_input(features=self.modulations)
            .add_input(features=self.features)
        )
        self.proj_o = (
            Linear(features=self.features, streams=streams)
            .add_input(features=self.modulations)
            .add_input(features=self.features)
        )

    def forward(self, modulation, stream=None):
        """
        Parameters
        ----------
        modulation : Tensor
            [N, S*I] -- stream is None
                or
            [N, I] -- stream is int
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            [N, S*M] -- stream is None
                or
            [N, M] -- stream is int
        """
        if self._past:
            assert self._past["stream"] == stream

            h = self._past["h"]
            c = self._past["c"]
        else:
            self._past["stream"] = stream

            features = self.features * self.streams if stream is None else self.features
            h = c = torch.zeros(modulation.size(0), features, device=self.device)

        if stream is None:
            modulation = modulation.repeat(1, self.streams)

        inputs = [modulation, h]

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
