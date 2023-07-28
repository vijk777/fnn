import torch
from .modules import Module
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
        common_channels,
        attention_channels,
        recurrent_channels,
        out_channels,
        groups=1,
        heads=1,
        spatial=3,
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
        spatial : int
            spatial kernel size
        dropout : float
            dropout probability -- [0, 1)
        """
        if recurrent_channels % groups != 0:
            raise ValueError("Recurrent channels must be divisible by groups")

        if attention_channels % heads != 0:
            raise ValueError("Attention channels must be divisible by heads")

        if heads < groups:
            raise ValueError("Heads cannot be less than groups")

        super().__init__()

        self.common_channels = int(common_channels)
        self.attention_channels = int(attention_channels)
        self.head_channels = int(attention_channels // heads)
        self.recurrent_channels = int(recurrent_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.heads = int(heads)
        self.spatial = int(spatial)
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

        self.common = Conv(
            in_channels=sum(self._inputs) + self.recurrent_channels,
            out_channels=self.common_channels,
            out_groups=self.groups,
            streams=self.streams,
            gain=None,
            bias=None,
        )

        self.conv = Conv(
            in_channels=self.common_channels,
            out_channels=self.common_channels,
            in_groups=self.groups,
            out_groups=self.groups,
            streams=self.streams,
            spatial=self.spatial,
            gain=None,
            bias=None,
        )

        def token(gain):
            return Conv(
                in_channels=self.common_channels,
                out_channels=self.attention_channels,
                in_groups=self.groups,
                out_groups=self.heads,
                streams=self.streams,
                gain=gain,
                bias=None,
            )

        self.proj_q = token(gain=self.head_channels**-0.5)
        self.proj_k = token(gain=None)
        self.proj_v = token(gain=None)

        def proj():
            return Conv(
                in_channels=self.common_channels + self.attention_channels,
                out_channels=self.recurrent_channels,
                in_groups=self.groups,
                out_groups=self.groups,
                streams=self.streams,
            )

        self.proj_z = proj()
        self.proj_h = proj()

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
            channels = self.streams * self.recurrent_channels
            groups = self.streams
        else:
            channels = self.recurrent_channels
            groups = 1

        if self.past:
            h = self.past["h"]
            h_drop = self.past["h_drop"]
        else:
            h = h_drop = torch.zeros(1, channels, 1, 1, device=self.device)

        c = cat_groups_2d([*x, h_drop], groups=groups, expand=True)
        c = self.common(c, stream=stream)
        c = self.conv(c, stream=stream)

        N, _, H, W = c.shape

        q = self.proj_q(c, stream=stream).view(N, groups, self.heads, self.head_channels, H * W)
        k = self.proj_k(c, stream=stream).view(N, groups, self.heads, self.head_channels, H * W)
        v = self.proj_v(c, stream=stream).view(N, groups, self.heads, self.head_channels, H * W)

        w = torch.einsum("N S G C Q , N S G C D -> N S G Q D", q, k).softmax(dim=-1)
        a = torch.einsum("N S G C D , N S G Q D -> N S G C Q", v, w).view(N, -1, H, W)

        ca = cat_groups_2d([c, a], groups=groups)

        z = torch.sigmoid(self.proj_z(ca, stream=stream))
        _h = torch.tanh(self.proj_h(ca, stream=stream))

        h = z * h + (1 - z) * _h
        h_drop = self.drop(h)

        self.past["h"] = h
        self.past["h_drop"] = h_drop

        return self.out(h_drop, stream=stream)


class CvtLstm(Recurrent):
    """Convolutional Transformer Lstm"""

    def __init__(
        self,
        recurrent_channels,
        attention_channels,
        common_channels,
        out_channels,
        groups=1,
        heads=1,
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
        attention_channels : int
            attention channels per stream
        out_channels : int
            out channels per stream
        groups : int
            groups per stream
        heads : int
            heads per stream
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

        if attention_channels % heads != 0:
            raise ValueError("Attention channels must be divisible by heads")

        if heads < groups:
            raise ValueError("Heads cannot be less than groups")

        super().__init__()

        self.recurrent_channels = int(recurrent_channels)
        self.attention_channels = int(attention_channels)
        self.head_channels = int(attention_channels // heads)
        self.common_channels = int(common_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.heads = int(heads)
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

        self.conv = Conv(
            in_channels=self.recurrent_channels * 2,
            out_channels=self.common_channels,
            in_groups=self.groups,
            out_groups=self.groups,
            streams=self.streams,
            spatial=self.spatial,
        )

        def token(gain):
            return Conv(
                in_channels=self.common_channels,
                out_channels=self.attention_channels,
                in_groups=self.groups,
                out_groups=self.heads,
                streams=self.streams,
                gain=gain,
                bias=None,
            )

        self.proj_q = token(gain=self.head_channels**-0.5)
        self.proj_k = token(gain=None)
        self.proj_v = token(gain=None)

        def proj(bias):
            return Conv(
                in_channels=self.common_channels + self.attention_channels,
                out_channels=self.recurrent_channels,
                in_groups=self.groups,
                out_groups=self.groups,
                streams=self.streams,
                bias=bias,
            )

        self.proj_i = proj(bias=self.init_input)
        self.proj_f = proj(bias=self.init_forget)
        self.proj_g = proj(bias=0)
        self.proj_o = proj(bias=0)

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
            channels = self.streams * self.recurrent_channels
            groups = self.streams
        else:
            channels = self.recurrent_channels
            groups = 1

        if self.past:
            h = self.past["h"]
            c = self.past["c"]
        else:
            h = c = torch.zeros(1, channels, 1, 1, device=self.device)

        if self.groups > 1:
            inputs = [h, *x]
        else:
            inputs = x

        x = torch.tanh(self.inputs(inputs, stream=stream))
        h = h.expand_as(x)
        xh = cat_groups_2d([x, h], groups=groups)

        z = self.conv(xh, stream=stream)
        N, _, H, W = z.shape

        q = self.proj_q(z, stream=stream).view(N, groups, self.heads, self.head_channels, H * W)
        k = self.proj_k(z, stream=stream).view(N, groups, self.heads, self.head_channels, H * W)
        v = self.proj_v(z, stream=stream).view(N, groups, self.heads, self.head_channels, H * W)

        w = torch.einsum("N S G C Q , N S G C D -> N S G Q D", q, k).softmax(dim=-1)
        a = torch.einsum("N S G C D , N S G Q D -> N S G C Q", v, w).view(N, -1, H, W)

        za = cat_groups_2d([z, a], groups=groups)

        i = torch.sigmoid(self.proj_i(za, stream=stream))
        f = torch.sigmoid(self.proj_f(za, stream=stream))
        g = torch.tanh(self.proj_g(za, stream=stream))
        o = torch.sigmoid(self.proj_o(za, stream=stream))

        c = f * c + i * g
        h = o * torch.tanh(c)
        h = self.drop(h)

        self.past["c"] = c
        self.past["h"] = h

        return self.out(h, stream=stream)


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
            channels = self.streams * self.recurrent_channels
            groups = self.streams
        else:
            channels = self.recurrent_channels
            groups = 1

        if self.past:
            h = self.past["h"]
            c = self.past["c"]
        else:
            h = c = torch.zeros(1, channels, 1, 1, device=self.device)

        if self.groups > 1:
            inputs = [h, *x]
        else:
            inputs = x

        x = torch.tanh(self.inputs(inputs, stream=stream))
        h = h.expand_as(x)
        xh = cat_groups_2d([x, h], groups=groups)

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
