import torch
from .modules import Module
from .elements import Conv, InterGroup, Accumulate, Dropout


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


class CvtLstm(Recurrent):
    """Convolutional Transformer Lstm"""

    def __init__(
        self,
        recurrent_channels,
        attention_channels,
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

        super().__init__()

        self.recurrent_channels = int(recurrent_channels)
        self.attention_channels = int(attention_channels)
        self.head_channels = int(attention_channels // heads)
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

        def conv(pad=None, gain=None):
            return Conv(
                in_channels=self.recurrent_channels,
                out_channels=self.attention_channels,
                in_groups=self.groups,
                out_groups=self.heads,
                streams=self.streams,
                spatial=self.spatial,
                pad=pad,
                gain=gain,
                bias=None,
            )

        self.q_xx = conv(pad="zeros", gain=self.head_channels**-0.5)
        self.q_xh = conv(pad="zeros", gain=self.head_channels**-0.5)
        self.q_hx = conv(pad="zeros", gain=self.head_channels**-0.5)
        self.q_hh = conv(pad="zeros", gain=self.head_channels**-0.5)

        self.k_xx = conv()
        self.k_xh = conv()
        self.k_hx = conv()
        self.k_hh = conv()

        self.v_xx = conv()
        self.v_xh = conv()
        self.v_hx = conv()
        self.v_hh = conv()

        def proj(bias):
            return Conv(
                in_channels=self.attention_channels,
                out_channels=self.recurrent_channels,
                in_groups=self.groups,
                out_groups=self.groups,
                streams=self.streams,
                gain=0.5,
                bias=bias,
            )

        def accumulate(bias=None):
            return Accumulate([proj(bias), proj(None), proj(None), proj(None)])

        self.proj_i = accumulate(self.init_input)
        self.proj_f = accumulate(self.init_forget)
        self.proj_g = accumulate(0)
        self.proj_o = accumulate(0)

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
            heads = self.streams * self.heads
        else:
            channels = self.recurrent_channels
            heads = self.heads

        if self.past:
            h = self.past["h"]
            c = self.past["c"]
        else:
            h = c = torch.zeros(1, channels, 1, 1, device=self.device)

        if self.groups > 1:
            x = self.inputs([h, *x], stream=stream)
        else:
            x = self.inputs(x, stream=stream)

        h = h.expand_as(x)
        N, _, H, W = h.shape
        attns = []

        for q_, k_, v_, query, token in [
            [self.q_xx, self.k_xx, self.v_xx, x, x],
            [self.q_xh, self.k_xh, self.v_xh, x, h],
            [self.q_hx, self.k_hx, self.v_hx, h, x],
            [self.q_hh, self.k_hh, self.v_hh, h, h],
        ]:
            q = q_(query, stream=stream).view(N, heads, self.head_channels, -1)
            k = k_(token, stream=stream).view(N, heads, self.head_channels, -1)
            v = v_(token, stream=stream).view(N, heads, self.head_channels, -1)

            w = torch.einsum("N G C Q , N G C D -> N G Q D", q, k).softmax(dim=3)
            a = torch.einsum("N G C D , N G Q D -> N G C Q", v, w).view(N, -1, H, W)
            attns.append(a)

        i = torch.sigmoid(self.proj_i(attns, stream=stream))
        f = torch.sigmoid(self.proj_f(attns, stream=stream))
        g = torch.tanh(self.proj_g(attns, stream=stream))
        o = torch.sigmoid(self.proj_o(attns, stream=stream))

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
                in_channels=self.recurrent_channels,
                out_channels=self.recurrent_channels,
                in_groups=self.groups,
                out_groups=self.groups,
                streams=self.streams,
                spatial=self.spatial,
                pad="zeros",
                gain=2**-0.5,
                bias=bias,
            )

        self.proj_i = Accumulate([conv(self.init_input), conv(None)])
        self.proj_f = Accumulate([conv(self.init_forget), conv(None)])
        self.proj_g = Accumulate([conv(0), conv(None)])
        self.proj_o = Accumulate([conv(0), conv(None)])

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
        if self.past:
            h = self.past["h"]
            c = self.past["c"]
        else:
            if stream is None:
                channels = self.streams * self.recurrent_channels
            else:
                channels = self.recurrent_channels
            h = c = torch.zeros(1, channels, 1, 1, device=self.device)

        if self.groups > 1:
            x = self.inputs([h, *x], stream=stream)
        else:
            x = self.inputs(x, stream=stream)

        h = h.expand_as(x)

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
