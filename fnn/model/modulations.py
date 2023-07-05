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
