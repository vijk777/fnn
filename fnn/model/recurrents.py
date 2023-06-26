import torch
from .parameters import Parameter, ParameterList
from .modules import Module
from .elements import Conv, StreamDropout
from .utils import to_groups_2d, cat_groups_2d


# -------------- Recurrent Prototype --------------


class Recurrent(Module):
    """Recurrent Module"""

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            [input channels per stream (I), ...]
        streams : int
            number of streams, S
        """
        raise NotImplementedError()

    @property
    def channels(self):
        """
        Returns
        -------
        int
            recurrent channels per stream (R)
        """
        raise NotImplementedError()

    def forward(self, inputs, stream=None):
        """
        Parameters
        ----------
        inputs : Sequence[Tensor]
            [[N, S*I, H, W] ...] -- stream is None
                or
            [[N, I, H, W] ...] -- stream is int
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, S*R, H//D, W//D] -- stream is None
                or
            [N, R, H//D, W//D] -- stream is int
        """
        raise NotImplementedError()


# -------------- Recurrent Types --------------


class Rvt(Recurrent):
    """Recurrent Vision Transformer"""

    def __init__(self, channels, groups, kernel_size, dropout=0):
        """
        Parameters
        ----------
        channels : int
            rvt channels per stream
        groups : int
            groups per stream
        kernel_size : int
            kernel size
        dropout : float
            dropout probability -- [0, 1)
        """
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")

        super().__init__()

        self._channels = int(channels)
        self.groups = int(groups)
        self.kernel_size = int(kernel_size)
        self._dropout = float(dropout)

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            [input channels per stream (I), ...]
        streams : int
            number of streams, S
        """
        self.inputs = list(map(int, inputs))
        self.streams = int(streams)

        self.proj_x = Conv(channels=self.channels, groups=self.groups, streams=self.streams)
        for _channels in inputs:
            self.proj_x.add_input(channels=_channels)

        if self.groups > 1:
            self.proj_x.add_intergroup()

        gain = (self.channels / self.groups) ** -0.5
        self.proj_q = Conv(channels=self.channels, groups=self.groups, streams=self.streams, init_gain=gain, bias=False)
        self.proj_q.add_input(
            channels=self.channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
        )
        self.proj_q.gains.decay = True

        self.proj_k = Conv(channels=self.channels, groups=self.groups, streams=self.streams, gain=False, bias=False)
        self.proj_k.add_input(
            channels=self.channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
            pad=False,
        )

        self.proj_v = Conv(channels=self.channels, groups=self.groups, streams=self.streams)
        self.proj_v.add_input(
            channels=self.channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
            pad=False,
        )

        self.proj_i = Conv(channels=self.channels, groups=self.groups, streams=self.streams)
        self.proj_i.add_input(
            channels=self.channels,
            groups=self.groups,
        )
        self.proj_i.add_input(
            channels=self.channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.proj_f = Conv(channels=self.channels, groups=self.groups, streams=self.streams)
        self.proj_f.add_input(
            channels=self.channels,
            groups=self.groups,
        )
        self.proj_f.add_input(
            channels=self.channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.proj_g = Conv(channels=self.channels, groups=self.groups, streams=self.streams)
        self.proj_g.add_input(
            channels=self.channels,
            groups=self.groups,
        )
        self.proj_g.add_input(
            channels=self.channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.proj_o = Conv(channels=self.channels, groups=self.groups, streams=self.streams)
        self.proj_o.add_input(
            channels=self.channels,
            groups=self.groups,
        )
        self.proj_o.add_input(
            channels=self.channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.drop = StreamDropout(p=self._dropout, streams=self.streams)

        self.past = [dict() for _ in range(self.streams + 1)]

    def _restart(self):
        self.dropout(p=self._dropout)

    def _reset(self):
        for past in self.past:
            past.clear()

    @property
    def channels(self):
        """
        Returns
        -------
        int
            recurrent channels per stream (R)
        """
        return self._channels

    def forward(self, inputs, stream=None):
        """
        Parameters
        ----------
        inputs : Sequence[Tensor]
            [[N, S*I, H, W] ...] -- stream is None
                or
            [[N, I, H, W] ...] -- stream is int
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, S*R, H//D, W//D] -- stream is None
                or
            [N, R, H//D, W//D] -- stream is int
        """
        if stream is None:
            past = self.past[self.streams]
            channels = self.channels * self.streams
            groups = self.groups * self.streams
        else:
            past = self.past[stream]
            channels = self.channels
            groups = self.groups

        if past:
            h = past["h"]
            c = past["c"]
        else:
            N, _, H, W = inputs[0].shape
            h = c = torch.zeros(N, channels, H, W, device=self.device)

        if self.groups > 1:
            x = self.proj_x([*inputs, h], stream=stream)
        else:
            x = self.proj_x(inputs, stream=stream)

        xh = cat_groups_2d([x, h], groups=groups)

        q = to_groups_2d(self.proj_q([xh], stream=stream), groups).flatten(3, 4)
        k = to_groups_2d(self.proj_k([xh], stream=stream), groups).flatten(3, 4)
        v = to_groups_2d(self.proj_v([xh], stream=stream), groups).flatten(3, 4)

        w = torch.einsum("N G C Q , N G C D -> N G Q D", q, k).softmax(dim=3)
        a = torch.einsum("N G C D , N G Q D -> N G C Q", v, w).view_as(x)

        i = torch.sigmoid(self.proj_i([a, xh], stream=stream))
        f = torch.sigmoid(self.proj_f([a, xh], stream=stream))
        g = torch.tanh(self.proj_g([a, xh], stream=stream))
        o = torch.sigmoid(self.proj_o([a, xh], stream=stream))

        c = f * c + i * g
        h = o * torch.tanh(c)
        h = self.drop(h, stream=stream)

        past["c"] = c
        past["h"] = h

        return h
