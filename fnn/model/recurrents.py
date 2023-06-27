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
            output channels per stream (O)
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
            [N, S*O, H, W] -- stream is None
                or
            [N, O, H, W] -- stream is int
        """
        raise NotImplementedError()


# -------------- Recurrent Types --------------


class Rvt(Recurrent):
    """Recurrent Vision Transformer"""

    def __init__(
        self,
        recurrent_channels,
        attention_channels,
        out_channels,
        groups=1,
        heads=1,
        kernel_size=3,
        dropout=0,
    ):
        """
        Parameters
        ----------
        recurrent_channels : int
            recurrent channels per stream
        attention_channels : int
            attention channels per stream
        out_channels : int
            out channels per stream
        groups : int
            groups per stream
        heads : int
            heads per stream
        kernel_size : int
            kernel size
        dropout : float
            dropout probability -- [0, 1)
        """
        if recurrent_channels % groups != 0:
            raise ValueError("channels must be divisible by groups")

        super().__init__()

        self.recurrent_channels = int(recurrent_channels)
        self.attention_channels = int(attention_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.heads = int(heads)
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

        self.drop_h = StreamDropout(p=self._dropout, streams=self.streams)

        self.proj_x = Conv(
            channels=self.recurrent_channels,
            groups=self.groups,
            streams=self.streams,
        )
        for _channels in inputs:
            self.proj_x.add_input(channels=_channels)

        if self.groups > 1:
            self.proj_x.add_intergroup()

        self.proj_q = Conv(
            channels=self.attention_channels,
            groups=self.heads,
            streams=self.streams,
            init_gain=(self.attention_channels / self.heads) ** -0.5,
            decay_gain=True,
            bias=False,
        )
        self.proj_q.add_input(
            channels=self.recurrent_channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.proj_k = Conv(
            channels=self.attention_channels,
            groups=self.heads,
            streams=self.streams,
            gain=False,
            bias=False,
        )
        self.proj_k.add_input(
            channels=self.recurrent_channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
            pad=False,
        )

        self.proj_v = Conv(
            channels=self.attention_channels,
            groups=self.heads,
            streams=self.streams,
        )
        self.proj_v.add_input(
            channels=self.recurrent_channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
            pad=False,
        )

        self.proj_i = Conv(
            channels=self.recurrent_channels,
            groups=self.groups,
            streams=self.streams,
        )
        self.proj_i.add_input(
            channels=self.attention_channels,
            groups=self.groups,
        )
        self.proj_i.add_input(
            channels=self.recurrent_channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.proj_f = Conv(
            channels=self.recurrent_channels,
            groups=self.groups,
            streams=self.streams,
        )
        self.proj_f.add_input(
            channels=self.attention_channels,
            groups=self.groups,
        )
        self.proj_f.add_input(
            channels=self.recurrent_channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.proj_g = Conv(
            channels=self.recurrent_channels,
            groups=self.groups,
            streams=self.streams,
        )
        self.proj_g.add_input(
            channels=self.attention_channels,
            groups=self.groups,
        )
        self.proj_g.add_input(
            channels=self.recurrent_channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.proj_o = Conv(
            channels=self.recurrent_channels,
            groups=self.groups,
            streams=self.streams,
        )
        self.proj_o.add_input(
            channels=self.attention_channels,
            groups=self.groups,
        )
        self.proj_o.add_input(
            channels=self.recurrent_channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
        )

        if self.recurrent_channels == self.out_channels and self.groups == 1:
            self.proj_y = None
        else:
            self.proj_y = Conv(
                channels=self.out_channels,
                streams=self.streams,
            )
            self.proj_y.add_input(
                channels=self.recurrent_channels,
            )

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
            output channels per stream (O)
        """
        return self.out_channels

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
            [N, S*O, H, W] -- stream is None
                or
            [N, O, H, W] -- stream is int
        """
        N, _, H, W = inputs[0].shape

        if stream is None:
            past = self.past[self.streams]
            channels = self.recurrent_channels * self.streams
            groups = self.groups * self.streams
            heads = self.heads * self.streams
        else:
            past = self.past[stream]
            channels = self.recurrent_channels
            groups = self.groups
            heads = self.heads

        if past:
            h = past["h"]
            c = past["c"]
        else:
            h = c = torch.zeros(N, channels, H, W, device=self.device)

        if self.groups > 1:
            x = self.proj_x([*inputs, h], stream=stream)
        else:
            x = self.proj_x(inputs, stream=stream)

        xh = cat_groups_2d([x, h], groups=groups)

        q = to_groups_2d(self.proj_q([xh], stream=stream), groups=heads).flatten(3, 4)
        k = to_groups_2d(self.proj_k([xh], stream=stream), groups=heads).flatten(3, 4)
        v = to_groups_2d(self.proj_v([xh], stream=stream), groups=heads).flatten(3, 4)

        w = torch.einsum("N G C Q , N G C D -> N G Q D", q, k).softmax(dim=3)
        a = torch.einsum("N G C D , N G Q D -> N G C Q", v, w).view(N, -1, H, W)

        i = torch.sigmoid(self.proj_i([a, xh], stream=stream))
        f = torch.sigmoid(self.proj_f([a, xh], stream=stream))
        g = torch.tanh(self.proj_g([a, xh], stream=stream))
        o = torch.sigmoid(self.proj_o([a, xh], stream=stream))

        c = f * c + i * g
        h = o * torch.tanh(c)
        h = self.drop_h(h, stream=stream)

        past["c"] = c
        past["h"] = h

        if self.proj_y is None:
            y = h
        else:
            y = self.proj_y([h], stream=stream)

        return y
