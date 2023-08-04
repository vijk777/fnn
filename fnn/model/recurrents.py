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
        attention_channels,
        projection_channels,
        recurrent_channels,
        out_channels,
        groups=1,
        heads=1,
        spatial=3,
        init_gate=1,
        dropout=0,
    ):
        """
        Parameters
        ----------
        attention_channels : int
            attention channels per stream
        projection_channels : int
            projection channels per stream
        recurrent_channels : int
            recurrent channels per stream
        out_channels : int
            out channels per stream
        groups : int
            groups per stream
        heads : int
            heads per stream
        spatial : int
            spatial kernel size
        init_gate : float
            initial gate bias
        dropout : float
            dropout probability -- [0, 1)
        """
        if heads % groups != 0:
            raise ValueError("Heads must be divisible by groups")

        if attention_channels % heads != 0:
            raise ValueError("Attention channels must be divisible by heads")

        if projection_channels % groups != 0:
            raise ValueError("Projection channels must be divisible by groups")

        if recurrent_channels % groups != 0:
            raise ValueError("Recurrent channels must be divisible by groups")

        super().__init__()

        self.attention_channels = int(attention_channels)
        self.head_channels = int(attention_channels // heads)
        self.projection_channels = int(projection_channels)
        self.recurrent_channels = int(recurrent_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.heads = int(heads)
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
                channels=self.recurrent_channels,
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
                out_channels=self.recurrent_channels,
                out_groups=self.groups,
                streams=self.streams,
                gain=gain,
                bias=None if i else 0,
            )
            inputs.append(conv)

        self.proj_x = Accumulate(inputs)

        def token(channels, pad):
            return Conv(
                in_channels=self.recurrent_channels * 2,
                out_channels=channels,
                in_groups=self.groups,
                out_groups=self.heads,
                streams=self.streams,
                spatial=self.spatial,
                pad=pad,
                gain=None,
                bias=None,
            )

        self.proj_q = token(channels=self.attention_channels, pad="zeros")
        self.proj_k = token(channels=self.attention_channels, pad=None)
        self.proj_v = token(channels=self.projection_channels, pad=None)

        def scale():
            return Parameter(torch.ones([self.heads, self.head_channels]))

        self.scales = ParameterList([scale() for _ in range(self.streams)])
        self.scales.decay = False
        self.scales.norm_dim = 1

        def proj(bias):
            return Conv(
                in_channels=self.projection_channels,
                out_channels=self.recurrent_channels,
                in_groups=self.groups,
                out_groups=self.groups,
                streams=self.streams,
                bias=bias,
            )

        self.proj_z = proj(bias=self.init_gate)
        self.proj_h = proj(bias=0)

        self.drop = Dropout(p=self._dropout)

        self.out = Conv(
            in_channels=self.recurrent_channels * 2,
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
            s = torch.stack(list(self.scales))
        else:
            S = 1
            s = self.scales[stream][None]

        if self.past:
            h = self.past["h"]
        else:
            h = torch.zeros(1, S * self.recurrent_channels, 1, 1, device=self.device)

        if self.groups > 1:
            x = torch.tanh(self.proj_x([h, *x], stream=stream))
        else:
            x = torch.tanh(self.proj_x(x, stream=stream))

        xh = cat_groups_2d([x, h], groups=S * self.groups, expand=True)
        N, _, H, W = xh.shape

        q = self.proj_q(xh, stream=stream).view(N, S, self.heads, self.head_channels, -1)
        k = self.proj_k(xh, stream=stream).view(N, S, self.heads, self.head_channels, -1)
        v = self.proj_v(xh, stream=stream).view(N, S, self.heads, self.head_channels, -1)

        q = q / q.norm(p=2, dim=3, keepdim=True)
        k = k / k.norm(p=2, dim=3, keepdim=True) * s

        w = torch.einsum("N S G C Q , N S G C D -> N S G Q D", q, k).softmax(dim=-1)
        a = torch.einsum("N S G C D , N S G Q D -> N S G C Q", v, w).view(N, -1, H, W)

        z = torch.sigmoid(self.proj_z(a, stream=stream))
        _h = torch.tanh(self.proj_h(a, stream=stream))

        h = z * h + (1 - z) * self.drop(_h)
        self.past["h"] = h

        xh = cat_groups_2d([x, h], groups=S * self.groups)
        return self.out(xh, stream=stream)


class ConvLstm(Recurrent):
    """Convolutional Lstm"""

    def __init__(
        self,
        recurrent_channels,
        out_channels,
        groups=1,
        spatial=3,
        init_input=-1,
        init_forget=1,
        dropout=0,
    ):
        """
        Parameters
        ----------
        recurrent_channels : int
            recurrent channels per stream
        out_channels : int
            out channels per stream
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
        if recurrent_channels % groups != 0:
            raise ValueError("Recurrent channels must be divisible by groups")

        super().__init__()

        self.recurrent_channels = int(recurrent_channels)
        self.out_channels = int(out_channels)
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
                channels=self.recurrent_channels,
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
                out_channels=self.recurrent_channels,
                out_groups=self.groups,
                streams=self.streams,
                gain=gain,
                bias=None if i else 0,
            )
            inputs.append(conv)

        self.inputs = Accumulate(inputs)

        def conv(bias):
            return Conv(
                in_channels=self.recurrent_channels * 2,
                out_channels=self.recurrent_channels,
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
            in_channels=self.recurrent_channels,
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
            h = self.past["h"]
            c = self.past["c"]
        else:
            h = c = torch.zeros(1, S * self.recurrent_channels, 1, 1, device=self.device)

        if self.groups > 1:
            x = torch.tanh(self.proj_x([h, *x], stream=stream))
        else:
            x = torch.tanh(self.proj_x(x, stream=stream))

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
