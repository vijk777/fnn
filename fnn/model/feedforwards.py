from torch.nn.functional import avg_pool2d
from .utils import cat_groups_2d
from .modules import Module, ModuleList
from .elements import Conv, StreamDropout, nonlinearity


# -------------- Feedforward Prototype --------------


class Feedforward(Module):
    """Feature Module"""

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            [input channels per stream (I), ...]
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
            feedforward channels per stream (F)
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
            [N, S*F, H//D, W//D] -- stream is None
                or
            [N, F, H//D, W//D] -- stream is int
        """
        raise NotImplementedError()


# -------------- Feedforward Types --------------


class Block(Module):
    """Dense Connection Block"""

    def __init__(self, channels, groups, layers, pool_size, kernel_size, dynamic_size, nonlinear=None):
        """
        Parameters
        ----------
        channels : int
            channels per stream (C)
        groups : int
            groups per stream
        layers : int
            number of layers
        pool_size : int
            pool size
        kernel_size : int
            kernel size
        dynamic_size : int
            dynamic size
        nonlinear : str | None
            nonlinearity
        """
        super().__init__()
        self.channels = int(channels)
        self.groups = int(groups)
        self.layers = int(layers)
        self.pool_size = int(pool_size)
        self.kernel_size = int(kernel_size)
        self.dynamic_size = int(dynamic_size)

        self.nonlinear, self.gamma = nonlinearity(nonlinear)

        if self.pool_size == 1:
            self.pool = lambda x: x
        else:
            self.pool = lambda x: [avg_pool2d(_, self.pool_size) * self.pool_size for _ in x]

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        inputs : Sequence[int]
            [input channels per stream (I), ...]
        streams : int
            number of streams (S)
        """
        self.inputs = list(map(int, inputs))
        self.streams = int(streams)

        def _conv():
            return Conv(channels=self.channels, streams=self.streams, groups=self.groups)

        def conn(l):
            if l:
                inputs = [self.channels] * (l + 1)

            elif len(self.inputs) > 1 or self.inputs[0] != self.channels:
                inputs = self.inputs

            else:
                return

            c = _conv()
            for channels in inputs:
                c.add_input(channels=channels)

            return c

        def conv():
            return _conv().add_input(
                channels=self.channels,
                groups=self.groups,
                kernel_size=self.kernel_size,
                dynamic_size=self.dynamic_size,
            )

        self.conns = ModuleList([conn(l) for l in range(self.layers)])
        self.convs = ModuleList([conv() for _ in range(self.layers)])

    def forward(self, x, stream=None):
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
        List[Tensor]
            [[N, S*C, H', W'], ... x (layers + 1)] -- stream is None
                or
            [[N, C, H', W'], ... x (layers + 1)] -- stream is int
        """
        x = self.pool(x)
        z = []

        for l, (conn, conv) in enumerate(zip(self.conns, self.convs)):

            if l == 0:
                if conn is None:
                    y = x[0]
                else:
                    y = conn(x, stream=stream)
                z.append(y)

            else:
                y = conn(z, stream=stream)

            y = conv([y], stream=stream)
            y = self.nonlinear(y) * self.gamma

            z.append(y)

        return z


class Dense(Feedforward):
    """Dense Connection Network"""

    def __init__(
        self,
        pre_kernel,
        pre_stride,
        block_channels,
        block_groups,
        block_layers,
        block_pools,
        block_kernels,
        block_dynamics,
        nonlinear=None,
        dropout=0,
    ):
        """
        Parameters
        ----------
        pre_kernel : int
            pre kernel size
        pre_stride : int
            pre stride
        block_channels : Sequence[int]
            block channels per stream
        block_groups : Sequence[int]
            block groups per stream
        block_layers : Sequence[int]
            block layers
        block_pools : Sequence[int]
            block pool sizes
        block_kernels : Sequence[int]
            block kernel sizes
        block_dynamics : Sequence[int]
            block dynamic sizes
        nonlinear : str | None
            nonlinearity
        dropout : float
            dropout probability -- [0, 1)
        """
        assert (
            len(block_channels)
            == len(block_groups)
            == len(block_layers)
            == len(block_pools)
            == len(block_kernels)
            == len(block_dynamics)
        )
        super().__init__()

        self.pre_kernel = int(pre_kernel)
        self.pre_stride = int(pre_stride)
        self.block_channels = list(map(int, block_channels))
        self.block_groups = list(map(int, block_groups))
        self.block_layers = list(map(int, block_layers))
        self.block_pools = list(map(int, block_pools))
        self.block_kernels = list(map(int, block_kernels))
        self.block_dynamics = list(map(int, block_dynamics))
        self.nonlinear = str(nonlinear)
        self._drop = float(dropout)

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        inputs : Sequence[int]
            [input channels per stream (I), ...]
        streams : int
            number of streams (S)
        """
        self.inputs = list(map(int, inputs))
        self.streams = int(streams)

        self.pre = Conv(channels=self.block_channels[0], groups=self.block_groups[0], streams=self.streams)
        for channels in self.inputs:
            self.pre.add_input(
                channels=channels,
                kernel_size=self.pre_kernel,
                stride=self.pre_stride,
                pad_mode="replicate",
            )

        self.blocks = ModuleList([])
        inputs = [self.block_channels[0]]

        for channels, groups, layers, pool, kernel, dynamic in zip(
            self.block_channels,
            self.block_groups,
            self.block_layers,
            self.block_pools,
            self.block_kernels,
            self.block_dynamics,
        ):
            block = Block(
                channels=channels,
                groups=groups,
                layers=layers,
                pool_size=pool,
                kernel_size=kernel,
                dynamic_size=dynamic,
                nonlinear=self.nonlinear,
            )
            block._init(
                inputs=inputs,
                streams=self.streams,
            )

            self.blocks.append(block)
            inputs = [channels] * (layers + 1)

        self.drop = StreamDropout(p=self._drop, streams=self.streams)

    def _restart(self):
        self.dropout(p=self._drop)

    @property
    def channels(self):
        """
        Returns
        -------
        int
            feedforward channels per stream (F)
        """
        return self.block_channels[-1] * (self.block_layers[-1] + 1)

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
            [N, S*F, H//D, W//D] -- stream is None
                or
            [N, F, H//D, W//D] -- stream is int
        """
        x = [self.pre(inputs, stream=stream)]

        for block in self.blocks:
            x = block(x, stream=stream)

        if stream is None:
            x = cat_groups_2d(x, groups=self.streams)
        else:
            x = cat_groups_2d(x, groups=1)

        return self.drop(x, stream=stream)
