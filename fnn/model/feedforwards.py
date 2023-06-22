import math
import torch
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
            number of streams, S
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
            specific stream | all streams

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
    """Dense Block"""

    def __init__(self, channels, groups, layers, pool_size, kernel_size, dynamic_size, nonlinear=None):
        """
        Parameters
        ----------
        channels : int
            dense channels per stream
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
        self.pool = None if self.pool_size == 1 else torch.nn.AvgPool2d(self.pool_size)

    def _init(self, mix, inputs, streams):
        """
        Parameters
        ----------
        mix : bool
            mix input channels
        inputs : int
            input channels per stream
        streams : int
            number of streams
        """
        self.mix = bool(mix)
        self.inputs = int(inputs)
        self.streams = int(streams)

        def new_conv():
            return Conv(channels=self.channels, streams=self.streams, groups=self.groups)

        def layer_conv(layer):
            conv = new_conv()

            for _ in range(layer):
                conv.add_input(
                    channels=self.channels,
                    groups=self.groups,
                )

            conv.add_input(
                channels=self.channels,
                groups=self.groups,
                kernel_size=self.kernel_size,
                dynamic_size=self.dynamic_size,
            )
            return conv

        if self.mix or self.inputs != self.channels:
            self.proj = new_conv().add_input(channels=self.inputs)
        else:
            self.proj = None

        self.convs = ModuleList([layer_conv(layer) for layer in range(self.layers)])

    def forward(self, x, stream=None):
        """
        Parameters
        ----------
        x : Tensor
            [N, S*I, H, W] -- stream is None
                or
            [N, I, H, W] -- stream is int
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            [N, S*F, H, W] -- stream is None
                or
            [N, F, H, W] -- stream is int
        """
        if self.pool is not None:
            x = self.pool(x)

        if self.proj is not None:
            x = self.proj([x], stream=stream)

        y = []
        for conv in self.convs:
            y.append(x)
            x = conv(y, stream=stream)
            x = self.nonlinear(x) * self.gamma

        return x


class Dense(Feedforward):
    """Dense Network"""

    def __init__(
        self,
        pre_channels,
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
        pre_channels : int
            pre channels per stream
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

        self.pre_channels = int(pre_channels)
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
            number of streams, S
        """
        self.inputs = list(map(int, inputs))
        self.streams = int(streams)

        self.pre = Conv(channels=self.pre_channels, groups=self.block_groups[0], streams=self.streams)
        for channels in self.inputs:
            self.pre.add_input(
                channels=channels,
                kernel_size=self.pre_kernel,
                stride=self.pre_stride,
            )

        self.blocks = ModuleList([])
        self.drops = ModuleList([])

        inputs = self.pre_channels
        mix = False

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
                mix=mix,
                inputs=inputs,
                streams=self.streams,
            )
            drop = StreamDropout(p=self._drop, streams=self.streams)

            self.blocks.append(block)
            self.drops.append(drop)

            inputs = channels
            mix = groups > 1

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
        return self.block_channels[-1]

    def forward(self, inputs, stream=None):
        """
        Parameters
        ----------
        inputs : Sequence[Tensor]
            [[N, S*I, H, W] ...] -- stream is None
                or
            [[N, I, H, W] ...] -- stream is int
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            [N, S*F, H//D, W//D] -- stream is None
                or
            [N, F, H//D, W//D] -- stream is int
        """
        x = self.pre(inputs, stream=stream)

        for block, drop in zip(self.blocks, self.drops):
            x = block(x, stream=stream)
            x = drop(x, stream=stream)

        return x
