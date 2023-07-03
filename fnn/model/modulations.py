import torch
from .modules import Module
from .elements import Linear, StreamFlatDropout, nonlinearity
from .utils import cat_groups


# -------------- Modulation Base --------------


class Modulation(Module):
    """Modulation Model"""

    def _init(self, modulations, streams):
        """
        Parameters
        ----------
        modulations : int
            modulation inputs per stream (I)
        streams : int
            number of streams (S)
        """
        raise NotImplementedError()

    @property
    def features(self):
        """
        Returns
        -------
        int
            modulation features per stream (M)
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
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, S*M] -- stream is None
                or
            [N, M] -- stream is int
        """
        raise NotImplementedError()


# -------------- Modulation Types --------------


class LnLstm(Modulation):
    """Linear -> Nonlinear -> Lstm"""

    def __init__(self, features, nonlinear=None, dropout=0):
        """
        Parameters
        ----------
        features : int
            feature size
        nonlinear : str | None
            nonlinearity
        dropout : float
            dropout probability -- [0, 1)
        """
        super().__init__()

        self._features = int(features)
        self.nonlinear, self.gamma = nonlinearity(nonlinear)
        self._dropout = float(dropout)

    def _init(self, modulations, streams):
        """
        Parameters
        ----------
        modulations : int
            modulation inputs per stream (I)
        streams : int
            number of streams (S)
        """
        self.modulations = int(modulations)
        self.streams = int(streams)

        self.drop_x = StreamFlatDropout(p=self._dropout, streams=self.streams)
        self.drop_h = StreamFlatDropout(p=self._dropout, streams=self.streams)

        def linear(inputs):
            return Linear(features=self.features, streams=self.streams).add_input(features=inputs)

        self.proj_x = linear(self.modulations)
        self.proj_i = linear(self.features * 2)
        self.proj_f = linear(self.features * 2)
        self.proj_g = linear(self.features * 2)
        self.proj_o = linear(self.features * 2)

        self.past = [dict() for _ in range(self.streams + 1)]

    def _restart(self):
        self.dropout(p=self._dropout)

    def _reset(self):
        for past in self.past:
            past.clear()

    @property
    def features(self):
        """
        Returns
        -------
        int
            modulation features per stream (M)
        """
        return self._features

    def forward(self, modulation, stream=None):
        """
        Parameters
        ----------
        modulation : Tensor
            [N, S*I] -- stream is None
                or
            [N, I] -- stream is int
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, S*M] -- stream is None
                or
            [N, M] -- stream is int
        """
        if stream is None:
            past = self.past[self.streams]
            features = self.features * self.streams
            groups = self.streams
        else:
            past = self.past[stream]
            features = self.features
            groups = 1

        if past:
            h = past["h"]
            c = past["c"]
        else:
            h = c = torch.zeros(1, features, device=self.device)

        x = self.proj_x([modulation], stream=stream)
        x = self.nonlinear(x) * self.gamma
        x = self.drop_x(x, stream=stream)

        xh = cat_groups([x, h.expand_as(x)], groups=groups)

        i = torch.sigmoid(self.proj_i([xh], stream=stream))
        f = torch.sigmoid(self.proj_f([xh], stream=stream))
        g = torch.tanh(self.proj_g([xh], stream=stream))
        o = torch.sigmoid(self.proj_o([xh], stream=stream))

        c = f * c + i * g
        h = o * torch.tanh(c)
        h = self.drop_h(h, stream=stream)

        past["c"] = c
        past["h"] = h

        return h


class FlatLstm(Modulation):
    """Flat Lstm"""

    def __init__(self, lstm_features, out_features, dropout=0):
        """
        Parameters
        ----------
        lstm_features : int
            lstm features per stream
        out_features : int
            out features per stream
        dropout : float
            dropout probability -- [0, 1)
        """
        super().__init__()

        self.lstm_features = int(lstm_features)
        self.out_features = int(out_features)
        self._dropout = float(dropout)

    def _init(self, modulations, streams):
        """
        Parameters
        ----------
        modulations : int
            modulation inputs per stream (I)
        streams : int
            number of streams (S)
        """
        self.modulations = int(modulations)
        self.streams = int(streams)

        self.drop = StreamFlatDropout(p=self._dropout, streams=self.streams)

        def linear(inputs, outputs, init_gain=1):
            return Linear(features=outputs, streams=self.streams, init_gain=init_gain).add_input(features=inputs)

        self.proj_i = linear(self.modulations + self.lstm_features, self.lstm_features)
        self.proj_f = linear(self.modulations + self.lstm_features, self.lstm_features)
        self.proj_g = linear(self.modulations + self.lstm_features, self.lstm_features)
        self.proj_o = linear(self.modulations + self.lstm_features, self.lstm_features)

        if self.lstm_features == self.out_features:
            self.proj_y = None
        else:
            self.proj_y = linear(self.lstm_features, self.out_features, 0)

        self.past = dict()

    def _restart(self):
        self.dropout(p=self._dropout)

    def _reset(self):
        self.past.clear()

    @property
    def features(self):
        """
        Returns
        -------
        int
            modulation features per stream (M)
        """
        return self.out_features

    def forward(self, modulation, stream=None):
        """
        Parameters
        ----------
        modulation : Tensor
            [N, S*I] -- stream is None
                or
            [N, I] -- stream is int
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, S*M] -- stream is None
                or
            [N, M] -- stream is int
        """
        if stream is None:
            features = self.lstm_features * self.streams
            groups = self.streams
        else:
            features = self.lstm_features
            groups = 1

        if self.past:
            assert self.past["stream"] == stream
            h = self.past["h"]
            c = self.past["c"]
        else:
            self.past["stream"] = stream
            h = c = torch.zeros(modulation.size(0), features, device=self.device)

        xh = cat_groups([modulation, h], groups=groups)

        i = torch.sigmoid(self.proj_i([xh], stream=stream))
        f = torch.sigmoid(self.proj_f([xh], stream=stream))
        g = torch.tanh(self.proj_g([xh], stream=stream))
        o = torch.sigmoid(self.proj_o([xh], stream=stream))

        c = f * c + i * g
        h = o * torch.tanh(c)
        h = self.drop(h, stream=stream)

        self.past["c"] = c
        self.past["h"] = h

        if self.proj_y is None:
            y = h
        else:
            y = self.proj_y([h], stream=stream)

        return y
