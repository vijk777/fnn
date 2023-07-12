import torch
from .modules import Module
from .elements import Linear, Accumulate, FlatDropout


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

    def __init__(self, lstm_features, out_features, init_input=-1, init_forget=1, dropout=0):
        """
        Parameters
        ----------
        lstm_features : int
            lstm features per stream
        out_features : int
            out features per stream
        init_input : float
            initial input gate bias
        init_forget : float
            initial forget gate bias
        dropout : float
            dropout probability -- [0, 1)
        """
        super().__init__()

        self.lstm_features = int(lstm_features)
        self.out_features = int(out_features)
        self.init_input = float(init_input)
        self.init_forget = float(init_forget)
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

        def linear(inputs, outputs, gain, bias):
            return Linear(
                in_features=inputs,
                out_features=outputs,
                streams=self.streams,
                gain=gain,
                bias=bias,
            )

        self.proj_i = Accumulate(
            [
                linear(self.modulations, self.lstm_features, 2**-0.5, self.init_input),
                linear(self.lstm_features, self.lstm_features, 2**-0.5, None),
            ]
        )
        self.proj_f = Accumulate(
            [
                linear(self.modulations, self.lstm_features, 2**-0.5, self.init_forget),
                linear(self.lstm_features, self.lstm_features, 2**-0.5, None),
            ]
        )
        self.proj_g = Accumulate(
            [
                linear(self.modulations, self.lstm_features, 2**-0.5, 0),
                linear(self.lstm_features, self.lstm_features, 2**-0.5, None),
            ]
        )
        self.proj_o = Accumulate(
            [
                linear(self.modulations, self.lstm_features, 2**-0.5, 0),
                linear(self.lstm_features, self.lstm_features, 2**-0.5, None),
            ]
        )

        self.drop = FlatDropout(p=self._dropout)

        self.out = linear(self.lstm_features, self.out_features, 0, 0)

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
        x = modulation

        if self.past:
            h = self.past["h"]
            c = self.past["c"]
        else:
            if stream is None:
                features = self.streams * self.lstm_features
            else:
                features = self.lstm_features
            h = c = torch.zeros([1, features], device=self.device)

        i = torch.sigmoid(self.proj_i([x, h], stream=stream))
        f = torch.sigmoid(self.proj_f([x, h], stream=stream))
        g = torch.tanh(self.proj_g([x, h], stream=stream))
        o = torch.sigmoid(self.proj_o([x, h], stream=stream))

        c = f * c + i * g
        h = o * torch.tanh(c)
        h = self.drop(h)

        self.past["c"] = c
        self.past["h"] = h

        return self.out(h, stream=stream)
