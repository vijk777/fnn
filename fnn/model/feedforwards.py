from torch.nn.functional import avg_pool2d
from .modules import Module, ModuleList
from .elements import Conv, Accumulate, Dropout, nonlinearity
from .utils import add


# -------------- Feedforward Base --------------


class Feedforward(Module):
    """Feature Module"""

    def _init(self, inputs, masks, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            [input channels per stream (I), ...]
        masks : Sequence[bool]
            initial mask for each input
        streams : int
            number of streams (S)
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
            [[N, I, H, W] ...] -- stream is int
                or
            [[N, S*I, H, W] ...] -- stream is None
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, O, H//D, W//D] -- stream is int
                or
            [N, S*O, H//D, W//D] -- stream is None
        """
        raise NotImplementedError()


# -------------- Feedforward Types --------------


class Block(Module):
    """Dense Connection Block"""

    def __init__(self, channels, groups, layers, temporal, spatial, pool, nonlinear=None):
        """
        Parameters
        ----------
        channels : int
            channels per stream (C)
        groups : int
            groups per stream
        layers : int
            number of layers
        temporal : int
            temporal kernel size
        spatial : int
            spatial kernel size
        pool_size : int
            spatial pooling size
        nonlinear : str | None
            nonlinearity
        """
        super().__init__()
        self.channels = int(channels)
        self.groups = int(groups)
        self._layers = int(layers)
        self.temporal = int(temporal)
        self.spatial = int(spatial)
        self.pool = int(pool)

        self.nonlinear, self.gamma = nonlinearity(nonlinear)

        if self.pool == 1:
            self.pool_fn = lambda x: x
        else:
            self.pool_fn = lambda x: avg_pool2d(x, self.pool) * self.pool

    def _init(self, streams):
        """
        Parameters
        ----------
        streams : int
            number of streams (S)
        """
        self.streams = int(streams)

        def conn():
            return Conv(
                in_channels=self.channels,
                in_groups=self.groups,
                out_channels=self.channels,
                out_groups=self.groups,
                streams=self.streams,
                gain=0,
                bias=None,
            )

        def conv():
            return Conv(
                in_channels=self.channels,
                out_channels=self.channels,
                in_groups=self.groups,
                out_groups=self.groups,
                streams=self.streams,
                temporal=self.temporal,
                spatial=self.spatial,
            )

        def layer(layer):
            modules = [conn() for _ in range(layer)] + [conv()]
            return Accumulate(modules)

        self.layers = ModuleList([layer(l) for l in range(self._layers)])

    def forward(self, x, stream=None):
        """
        Parameters
        ----------
        inputs : 4D Tensor
            [N, C, H, W] -- stream is int
                or
            [N, S*C, H, W] -- stream is None
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        List[4D Tensor]
            [[N, C, H', W'], ... x (layers + 1)] -- stream is int
                or
            [[N, S*C, H', W'], ... x (layers + 1)] -- stream is None
        """
        x = [self.pool_fn(x)]

        for layer in self.layers:

            y = layer(x, stream=stream)
            y = self.nonlinear(y) * self.gamma

            x.append(y)

        return x


class Dense(Feedforward):
    """Dense Connection Network"""

    def __init__(
        self,
        in_spatial,
        in_stride,
        block_channels,
        block_groups,
        block_layers,
        block_temporals,
        block_spatials,
        block_pools,
        out_channels,
        nonlinear=None,
        dropout=0,
    ):
        """
        Parameters
        ----------
        in_spatial : int
            input spatial kernel size
        in_stride : int
            input spatial stride
        block_channels : Sequence[int]
            block channels per stream
        block_groups : Sequence[int]
            block groups per stream
        block_layers : Sequence[int]
            block layers
        block_temporals : Sequence[int]
            block temporal kernel sizes
        block_spatials : Sequence[int]
            block spatial kernel sizes
        block_pools : Sequence[int]
            block spatial pooling sizes
        out_channels : int
            output channels per strem
        nonlinear : str | None
            nonlinearity
        dropout : float
            dropout probability -- [0, 1)
        """
        assert (
            len(block_channels)
            == len(block_groups)
            == len(block_layers)
            == len(block_temporals)
            == len(block_spatials)
            == len(block_pools)
        )
        super().__init__()

        self.in_spatial = int(in_spatial)
        self.in_stride = int(in_stride)
        self.block_channels = list(map(int, block_channels))
        self.block_groups = list(map(int, block_groups))
        self.block_layers = list(map(int, block_layers))
        self.block_temporals = list(map(int, block_temporals))
        self.block_spatials = list(map(int, block_spatials))
        self.block_pools = list(map(int, block_pools))
        self.out_channels = int(out_channels)
        self.nonlinear = str(nonlinear)
        self._drop = float(dropout)

    def _init(self, inputs, masks, streams):
        """
        Parameters
        ----------
        inputs : Sequence[int]
            [input channels per stream (I), ...]
        masks : Sequence[bool]
            initial mask for each input
        streams : int
            number of streams (S)
        """
        self._inputs = list(map(int, inputs))
        self.masks = list(map(bool, masks))
        self.streams = int(streams)

        assert len(self._inputs) == len(self.masks)
        assert sum(masks) > 0

        def proj(in_channels, out_channels, out_groups, spatial, stride, gain, bias):
            return Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                out_groups=out_groups,
                spatial=spatial,
                stride=stride,
                gain=gain,
                pad="replicate",
                streams=self.streams,
                bias=bias,
            )

        def accumulate(inputs, masks, out_channels, out_groups, spatial, stride):
            modules = []
            gain = sum(masks) ** -0.5
            for l, (i, m) in enumerate(zip(inputs, masks)):
                bias = None if l else 0
                module = proj(i, out_channels, out_groups, spatial, stride, gain * m, bias)
                modules.append(module)
            return Accumulate(modules)

        self.inputs = ModuleList()
        self.blocks = ModuleList()

        in_spatial = self.in_spatial
        in_stride = self.in_stride
        inputs = self._inputs
        masks = self.masks

        for channels, groups, layers, temporal, spatial, pool in zip(
            self.block_channels,
            self.block_groups,
            self.block_layers,
            self.block_temporals,
            self.block_spatials,
            self.block_pools,
        ):
            inputs = accumulate(
                inputs=inputs,
                masks=masks,
                out_channels=channels,
                out_groups=groups,
                spatial=in_spatial,
                stride=in_stride,
            )
            self.inputs.append(inputs)

            block = Block(
                channels=channels,
                groups=groups,
                layers=layers,
                temporal=temporal,
                spatial=spatial,
                pool=pool,
                nonlinear=self.nonlinear,
            )
            block._init(streams=self.streams)
            self.blocks.append(block)

            in_spatial = 1
            in_stride = 1
            inputs = (layers + 1) * [channels]
            masks = layers * [False] + [True]

        self.drop = ModuleList([Dropout(p=self._drop) for _ in inputs])

        self.out = accumulate(
            inputs=inputs,
            masks=masks,
            out_channels=self.out_channels,
            out_groups=1,
            spatial=1,
            stride=1,
        )

    def _restart(self):
        self.dropout(p=self._drop)

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
            [N, O, H', W'] -- stream is int
                or
            [N, S*O, H', W'] -- stream is None
        """
        for inp, block in zip(self.inputs, self.blocks):

            x = inp(x, stream=stream)
            x = block(x, stream=stream)

        x = [drop(_) for drop, _ in zip(self.drop, x)]

        return self.out(x, stream=stream)
