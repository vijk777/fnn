import torch
import math

from .containers import Module, ModuleList
from .elements import Conv, nonlinearity


class Feedforward(Module):
    @property
    def channels(self):
        raise NotImplementedError()

    @property
    def streams(self):
        raise NotImplementedError()

    @property
    def scale(self):
        raise NotImplementedError()

    def add_input(self, channels):
        """
        Parameters
        ----------
        channels : int
            input channels
        """
        raise NotImplementedError()

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : Sequence[Tensor]
            shapes = [n, c, h, w] or broadcastable

        Returns
        -------
        Tensor
            shape = [n, c', h // scale, w // scale]
        """
        raise NotImplementedError()


class Res3d(Feedforward):
    def __init__(self, channels, kernel_sizes, strides, streams, nonlinear=None):
        """
        Parameters
        ----------
        channels : Sequence[int]
            layer channels
        kernel_sizes : Sequence[int]
            layer kernel sizes
        strides : Sequence[int]
            layer strides
        streams : int
            number of streams
        nonlinear : str | None
            nonlinearity
        """
        assert len(channels) == len(kernel_sizes) == len(strides)
        super().__init__()

        self._channels = list(map(int, channels))
        self.kernel_sizes = list(map(int, kernel_sizes))
        self.strides = list(map(int, strides))
        self._streams = int(streams)

        self.conv = ModuleList([Conv(channels=channels[0], streams=streams)])
        self.res = ModuleList([Conv(channels=channels[0], streams=streams)])

        _channels = channels[0]
        for channels, size, stride in zip(
            channels[1:],
            kernel_sizes[1:],
            strides[1:],
        ):
            conv = Conv(channels=channels, streams=streams).add_input(
                channels=_channels,
                kernel_size=size,
                dynamic_size=size,
                stride=stride,
            )
            res = Conv(channels=channels, streams=streams).add_input(
                channels=_channels,
                kernel_size=stride,
                stride=stride,
            )
            self.conv.append(conv)
            self.res.append(res)
            _channels = channels

        for res in self.res:
            for gain in res.gains:
                torch.nn.init.constant_(gain, 0)

        self.nonlinear, self.gamma = nonlinearity(nonlinear)

    @property
    def channels(self):
        return self._channels[-1]

    @property
    def streams(self):
        return self._streams

    @property
    def scale(self):
        return math.prod(self.strides)

    def add_input(self, channels):
        """
        Parameters
        ----------
        channels : int
            input channels
        """
        self.conv[0].add_input(
            channels=channels,
            kernel_size=self.kernel_sizes[0],
            dynamic_size=self.kernel_sizes[0],
            stride=self.strides[0],
        )
        self.res[0].add_input(
            channels=channels,
            kernel_size=self.strides[0],
            stride=self.strides[0],
        )

    def forward(self, inputs, stream=None):
        """
        Parameters
        ----------
        inputs : Sequence[Tensor]
            shapes = [n, c, h, w] or broadcastable -- stream is None
                or
            shapes = [n, c // s, h, w] or broadcastable -- stream is not None
        stream : int | None
            specific stream index (int) or all streams (None)

        Returns
        -------
        Tensor
            shape = [n, c', h // scale, w // scale] -- stream is None
                or
            shape = [n, c' // s, h // scale, w // scale] -- stream is not None
        """
        for conv, res in zip(self.conv, self.res):

            c = conv(inputs, stream=stream)
            r = res(inputs, stream=stream)

            inputs = [self.nonlinear(c) * self.gamma + r]

        return inputs[0]
