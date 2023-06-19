import math
from torch.nn import Identity, AvgPool2d
from .modules import Module, ModuleList
from .elements import Conv, StreamDropout, nonlinearity
from .utils import to_groups_2d


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

    def __init__(self, channels, groups, layers, kernel_size, dynamic_size, nonlinear=None):
        """
        Parameters
        ----------
        channels : int
            dense channels per stream
        groups : int
            groups per stream
        layers : int
            number of layers
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
        self.kernel_size = int(kernel_size)
        self.dynamic_size = int(dynamic_size)

        self.nonlinear, self.gamma = nonlinearity(nonlinear)

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        inputs : int
            input channels per stream
        streams : int
            number of streams
        """
        self.inputs = int(inputs)
        self.streams = int(streams)

        self.conv = ModuleList([])
        self.dense = ModuleList([])

        channels = [self.inputs]
        for _ in range(self.layers):

            conv = Conv(channels=self.channels, streams=self.streams, groups=self.groups).add_input(
                channels=channels[-1],
                kernel_size=self.kernel_size,
                dynamic_size=self.dynamic_size,
            )
            channels.append(self.channels)

            dense = Conv(channels=self.channels, streams=self.streams)
            for _channels in channels:
                dense.add_input(channels=_channels)

            self.conv.append(conv)
            self.dense.append(dense)

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
        y = [x]

        for conv, dense in zip(self.conv, self.dense):

            x = conv([x], stream=stream)
            x = self.nonlinear(x) * self.gamma

            y.append(x)
            x = dense(y, stream=stream)

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
        block_kernels,
        block_dynamics,
        pool_sizes,
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
        block_kernels : Sequence[int]
            block kernel sizes
        block_dynamics : Sequence[int]
            block dynamic sizes
        pool_sizes : Sequence[int]
            pooling sizes
        nonlinear : str | None
            nonlinearity
        dropout : float
            dropout probability -- [0, 1)
        """
        assert (
            len(block_channels)
            == len(block_groups)
            == len(block_layers)
            == len(block_kernels)
            == len(block_dynamics)
            == len(pool_sizes)
        )
        super().__init__()

        self.pre_channels = int(pre_channels)
        self.pre_kernel = int(pre_kernel)
        self.pre_stride = int(pre_stride)

        self.block_channels = list(map(int, block_channels))
        self.block_groups = list(map(int, block_groups))
        self.block_layers = list(map(int, block_layers))
        self.block_kernels = list(map(int, block_kernels))
        self.block_dynamics = list(map(int, block_dynamics))
        self.pool_sizes = list(map(int, pool_sizes))

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

        self.pre = Conv(channels=self.pre_channels, streams=self.streams).add_input(
            channels=sum(self.inputs),
            kernel_size=self.pre_kernel,
            stride=self.pre_stride,
        )

        self.blocks = ModuleList([])
        self.pools = ModuleList([])

        inputs = self.pre_channels
        for channels, groups, layers, kernel, dynamic, pool in zip(
            self.block_channels,
            self.block_groups,
            self.block_layers,
            self.block_kernels,
            self.block_dynamics,
            self.pool_sizes,
        ):
            block = Block(
                channels=channels,
                groups=groups,
                layers=layers,
                kernel_size=kernel,
                dynamic_size=dynamic,
                nonlinear=self.nonlinear,
            )
            block._init(
                inputs=inputs,
                streams=self.streams,
            )

            pool = Identity() if pool == 1 else AvgPool2d(pool)

            self.blocks.append(block)
            self.pools.append(pool)
            inputs = channels

        self.drop = StreamDropout(p=self._drop, streams=self.streams)

    def _restart(self):
        self.drop.p = self._drop

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
        if len(self.inputs) == 1:
            x = inputs[0]
        elif stream is None:
            x = torch.cat([to_groups_2d(_, self.streams) for _ in inputs], 2).flatten(1, 2)
        else:
            x = torch.cat(inputs, 1)

        x = self.pre([x], stream=stream)

        for block, pool in zip(self.blocks, self.pools):
            x = block(x, stream=stream)
            x = pool(x)

        return self.drop(x, stream=stream)
