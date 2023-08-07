import torch
from .modules import Module
from .elements import Linear, FlatDropout
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

    def __init__(self, in_features, out_features, hidden_features, init_input=-1, init_forget=1, dropout=0):
        """
        Parameters
        ----------
        in_features : int
            in features per stream
        out_features : int
            out features per stream
        hidden_features : int
            hidden features per stream
        init_input : float
            initial input gate bias
        init_forget : float
            initial forget gate bias
        dropout : float
            dropout probability -- [0, 1)
        """
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.hidden_features = int(hidden_features)

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

        self.drop_x = FlatDropout(p=self._dropout)
        self.drop_h = FlatDropout(p=self._dropout)

        self.proj_x = Linear(
            in_features=self.modulations,
            out_features=self.in_features,
            streams=self.streams,
        )

        def linear(bias):
            return Linear(
                in_features=self.in_features + self.hidden_features,
                out_features=self.hidden_features,
                streams=self.streams,
                bias=bias,
            )

        self.proj_i = linear(bias=self.init_input)
        self.proj_f = linear(bias=self.init_forget)
        self.proj_g = linear(bias=0)
        self.proj_o = linear(bias=0)

        self.out = Linear(
            in_features=self.hidden_features,
            out_features=self.out_features,
            streams=self.streams,
            gain=0,
        )
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
            S = self.streams
        else:
            S = 1

        if self.past:
            h = self.past["h"]
            c = self.past["c"]
        else:
            h = c = torch.zeros([1, S * self.hidden_features], device=self.device)

        x = torch.tanh(self.proj_x(modulation, stream=stream))
        x = self.drop_x(x)

        xh = cat_groups([x, h], groups=S, expand=True)

        i = torch.sigmoid(self.proj_i(xh, stream=stream))
        f = torch.sigmoid(self.proj_f(xh, stream=stream))
        g = torch.tanh(self.proj_g(xh, stream=stream))
        o = torch.sigmoid(self.proj_o(xh, stream=stream))

        c = f * c + i * g
        h = o * torch.tanh(c)
        h = self.drop_h(h)

        self.past["c"] = c
        self.past["h"] = h

        return self.out(h, stream=stream)
