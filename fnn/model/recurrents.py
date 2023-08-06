import torch
from .modules import Module
from .parameters import Parameter, ParameterList
from .elements import Conv, InterGroup, Accumulate, Dropout
from .utils import cat_groups_2d


# -------------- Recurrent Base --------------


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

    def forward(self, x, stream=None):
        """
        Parameters
        ----------
        x : Sequence[Tensor]
            [[N, I, H, W] ...] -- stream is int
                or
            [[N, S*I, H, W] ...] -- stream is None
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, O, H, W] -- stream is int
                or
            [N, S*O, H, W] -- stream is None
        """
        raise NotImplementedError()


# -------------- Recurrent Types --------------


class Rvt(Recurrent):
    """Recurrent Vision Transformer"""

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        attention_channels,
        projection_channels,
        groups=1,
        spatial=3,
        init_gate=1,
        dropout=0,
    ):
        """
        Parameters
        ----------
        in_channels : int
            in channels per stream
        out_channels : int
            out channels per stream
        hidden_channels : int
            hidden channels per stream
        attention_channels : int
            attention channels per stream
        projection_channels : int
            projection channels per stream
        groups : int
            groups per stream
        spatial : int
            spatial kernel size
        init_gate : float
            initial gate bias
        dropout : float
            dropout probability -- [0, 1)
        """
        if in_channels % groups != 0:
            raise ValueError("Input channels must be divisible by groups")

        if hidden_channels % groups != 0:
            raise ValueError("Hidden channels must be divisible by groups")

        if attention_channels % groups != 0:
            raise ValueError("Attention channels must be divisible by groups")

        if projection_channels % groups != 0:
            raise ValueError("Projection channels must be divisible by groups")

        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.attention_channels = int(attention_channels)
        self.projection_channels = int(projection_channels)

        self.attn_channels = int(attention_channels // groups)
        self.proj_channels = int(projection_channels // groups)

        self.groups = int(groups)
        self.spatial = int(spatial)
        self.init_gate = float(init_gate)
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
        self._inputs = list(map(int, inputs))
        self.streams = int(streams)

        if self.groups > 1:
            gain = (len(self._inputs) + 1) ** -0.5
            intergroup = InterGroup(
                in_channels=self.hidden_channels,
                out_channels=self.in_channels,
                groups=self.groups,
                streams=self.streams,
                gain=gain,
            )
            inputs = [intergroup]
        else:
            gain = len(self._inputs) ** -0.5
            inputs = []

        for i, in_channels in enumerate(self._inputs):
            conv = Conv(
                in_channels=in_channels,
                out_channels=self.in_channels,
                out_groups=self.groups,
                streams=self.streams,
                gain=gain,
                bias=None if i else 0,
            )
            inputs.append(conv)

        self.proj_x = Accumulate(inputs)

        def token(in_channels, out_channels, pad, gain):
            return Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                in_groups=self.groups,
                out_groups=self.groups,
                streams=self.streams,
                spatial=self.spatial,
                pad=pad,
                gain=gain,
                bias=None,
            )

        self.proj_q_x = token(
            in_channels=self.in_channels,
            out_channels=self.attention_channels,
            pad="zeros",
            gain=None,
        )
        self.proj_q_c = token(
            in_channels=self.hidden_channels,
            out_channels=self.attention_channels,
            pad="zeros",
            gain=None,
        )

        self.proj_k_x = token(
            in_channels=self.in_channels,
            out_channels=self.attention_channels,
            pad=None,
            gain=self.attn_channels**-0.5,
        )
        self.proj_k_c = token(
            in_channels=self.hidden_channels,
            out_channels=self.attention_channels,
            pad=None,
            gain=self.attn_channels**-0.5,
        )

        self.proj_v_x = token(
            in_channels=self.in_channels,
            out_channels=self.projection_channels,
            pad=None,
            gain=None,
        )
        self.proj_v_c = token(
            in_channels=self.hidden_channels,
            out_channels=self.projection_channels,
            pad=None,
            gain=None,
        )

        def proj(channels, bias):
            return Conv(
                in_channels=channels,
                out_channels=self.hidden_channels,
                in_groups=self.groups,
                out_groups=self.groups,
                streams=self.streams,
                bias=bias,
            )

        self.proj_z = proj(
            channels=self.projection_channels * 2,
            bias=self.init_gate,
        )
        self.proj_n = proj(
            channels=self.projection_channels * 2,
            bias=0,
        )
        self.proj_h = proj(
            channels=self.in_channels + self.projection_channels * 2,
            bias=0,
        )

        self.drop = Dropout(p=self._dropout)

        self.out = Conv(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            streams=self.streams,
        )

        self.past = dict()

    def _restart(self):
        self.dropout(p=self._dropout)

    def _reset(self):
        self.past.clear()

    @property
    def channels(self):
        """
        Returns
        -------
        int
            output channels per stream (O)
        """
        return self.out_channels

    def forward(self, x, stream=None):
        """
        Parameters
        ----------
        x : Sequence[Tensor]
            [[N, I, H, W] ...] -- stream is int
                or
            [[N, S*I, H, W] ...] -- stream is None
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, O, H, W] -- stream is int
                or
            [N, S*O, H, W] -- stream is None
        """
        if stream is None:
            S = self.streams
        else:
            S = 1

        if self.past:
            c = self.past["c"]
            h = self.past["h"]
        else:
            h = c = torch.randn(1, S * self.hidden_channels, 1, 1, device=self.device)

        if self.groups > 1:
            x = self.proj_x([h, *x], stream=stream)
        else:
            x = self.proj_x(x, stream=stream)

        N, _, H, W = x.shape
        c = c.expand(N, -1, H, W)

        q_x = self.proj_q_x(x, stream=stream).view(N, S, self.groups, self.attn_channels, -1)
        q_c = self.proj_q_c(c, stream=stream).view(N, S, self.groups, self.attn_channels, -1)

        k_x = self.proj_k_x(x, stream=stream).view(N, S, self.groups, self.attn_channels, -1)
        k_c = self.proj_k_c(c, stream=stream).view(N, S, self.groups, self.attn_channels, -1)

        v_x = self.proj_v_x(x, stream=stream).view(N, S, self.groups, self.proj_channels, -1)
        v_c = self.proj_v_c(c, stream=stream).view(N, S, self.groups, self.proj_channels, -1)

        q = torch.stack([q_x, q_c], dim=0).unsqueeze(dim=4)
        k = torch.stack([k_x, k_c], dim=3).unsqueeze(dim=0)
        v = torch.stack([v_x, v_c], dim=3).unsqueeze(dim=0)

        q = q.expand(2, N, S, self.groups, 2, self.attn_channels, -1)
        k = k.expand(2, N, S, self.groups, 2, self.attn_channels, -1)
        v = v.expand(2, N, S, self.groups, 2, self.proj_channels, -1)

        w = torch.einsum("A N S G B C Q , A N S G B C D -> A N S G B Q D", q, k).softmax(dim=-1)
        a = torch.einsum("A N S G B C D , A N S G B Q D -> A N S G B C Q", v, w).view(2, N, -1, H, W)

        a_x, a_c = a.unbind(0)

        z = torch.sigmoid(self.proj_z(a_c, stream=stream))
        n = torch.tanh(self.proj_n(a_c, stream=stream))
        c = z * c + (1 - z) * n

        xa = cat_groups_2d([x, a_x], groups=S * self.groups)
        h = torch.tanh(self.proj_h(xa, stream=stream))
        h = self.drop(h)

        self.past["c"] = c
        self.past["h"] = h

        return self.out(h, stream=stream)


class ConvLstm(Recurrent):
    """Convolutional Lstm"""

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        groups=1,
        spatial=3,
        init_input=-1,
        init_forget=1,
        dropout=0,
    ):
        """
        Parameters
        ----------
        in_channels : int
            in channels per stream
        out_channels : int
            out channels per stream
        hidden_channels : int
            hidden channels per stream
        groups : int
            groups per stream
        spatial : int
            spatial kernel size
        init_input : float
            initial input gate bias
        init_forget : float
            initial forget gate bias
        dropout : float
            dropout probability -- [0, 1)
        """
        if in_channels % groups != 0:
            raise ValueError("Input channels must be divisible by groups")

        if hidden_channels % groups != 0:
            raise ValueError("Hidden channels must be divisible by groups")

        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)

        self.groups = int(groups)
        self.spatial = int(spatial)
        self.init_input = float(init_input)
        self.init_forget = float(init_forget)
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
        self._inputs = list(map(int, inputs))
        self.streams = int(streams)

        if self.groups > 1:
            gain = (len(self._inputs) + 1) ** -0.5
            intergroup = InterGroup(
                in_channels=self.hidden_channels,
                out_channels=self.in_channels,
                groups=self.groups,
                streams=self.streams,
                gain=gain,
            )
            inputs = [intergroup]
        else:
            gain = len(self._inputs) ** -0.5
            inputs = []

        for i, in_channels in enumerate(self._inputs):
            conv = Conv(
                in_channels=in_channels,
                out_channels=self.in_channels,
                out_groups=self.groups,
                streams=self.streams,
                gain=gain,
                bias=None if i else 0,
            )
            inputs.append(conv)

        self.proj_x = Accumulate(inputs)

        def conv(bias):
            return Conv(
                in_channels=self.in_channels + self.hidden_channels,
                out_channels=self.hidden_channels,
                in_groups=self.groups,
                out_groups=self.groups,
                streams=self.streams,
                spatial=self.spatial,
                bias=bias,
            )

        self.proj_i = conv(bias=self.init_input)
        self.proj_f = conv(bias=self.init_forget)
        self.proj_g = conv(bias=0)
        self.proj_o = conv(bias=0)

        self.drop = Dropout(p=self._dropout)

        self.out = Conv(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            streams=self.streams,
        )

        self.past = dict()

    def _restart(self):
        self.dropout(p=self._dropout)

    def _reset(self):
        self.past.clear()

    @property
    def channels(self):
        """
        Returns
        -------
        int
            output channels per stream (O)
        """
        return self.out_channels

    def forward(self, x, stream=None):
        """
        Parameters
        ----------
        x : Sequence[Tensor]
            [[N, I, H, W] ...] -- stream is int
                or
            [[N, S*I, H, W] ...] -- stream is None
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, O, H, W] -- stream is int
                or
            [N, S*O, H, W] -- stream is None
        """
        if stream is None:
            S = self.streams
        else:
            S = 1

        if self.past:
            c = self.past["c"]
            h = self.past["h"]
        else:
            h = c = torch.zeros(1, S * self.hidden_channels, 1, 1, device=self.device)

        if self.groups > 1:
            x = self.proj_x([h, *x], stream=stream)
        else:
            x = self.proj_x(x, stream=stream)

        xh = cat_groups_2d([x, h], groups=S * self.groups, expand=True)

        i = torch.sigmoid(self.proj_i(xh, stream=stream))
        f = torch.sigmoid(self.proj_f(xh, stream=stream))
        g = torch.tanh(self.proj_g(xh, stream=stream))
        o = torch.sigmoid(self.proj_o(xh, stream=stream))

        c = f * c + i * g
        h = o * torch.tanh(c)
        h = self.drop(h)

        self.past["c"] = c
        self.past["h"] = h

        return self.out(h, stream=stream)
