import torch
import math

from .containers import Module, ModuleList
from .elements import Conv, nonlinearity


class Feedforward(Module):
    @property
    def channels(self):
        """
        Returns
        -------
        int
            output channels per stream, c'
        """
        raise NotImplementedError()

    @property
    def scale(self):
        """
        Returns
        -------
        int
            scale of spatial downsampling, d
        """
        raise NotImplementedError()

    def init(self, channels, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            input channels per stream, c
        streams : int
            number of streams, s
        """
        raise NotImplementedError()

    def forward(self, inputs, stream=None):
        """
        Parameters
        ----------
        inputs : Sequence[Tensor]
            shapes = [n, c * s, h, w] -- stream is None
                or
            shapes = [n, c, h, w] -- stream is int
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            shape = [n, c', h // d, w // d] -- stream is None
                or
            shape = [n, c' * s, h // d, w // d] -- stream is int
        """
        raise NotImplementedError()


class Res3d(Feedforward):
    def __init__(self, channels, kernel_sizes, strides, nonlinear=None):
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
        self.nonlinear, self.gamma = nonlinearity(nonlinear)

    @property
    def channels(self):
        """
        Returns
        -------
        int
            output channels per stream, c'
        """
        return self._channels[-1]

    @property
    def scale(self):
        """
        Returns
        -------
        int
            scale of spatial downsampling, d
        """
        return math.prod(self.strides)

    def init(self, channels, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            input channels per stream, c
        streams : int
            number of streams, s
        """

        conv = Conv(channels=self._channels[0], streams=streams)
        res = Conv(channels=self._channels[0], streams=streams)

        for _channels in channels:
            conv.add_input(
                channels=_channels,
                kernel_size=self.kernel_sizes[0],
                dynamic_size=self.kernel_sizes[0],
                stride=self.strides[0],
            )
            res.add_input(
                channels=_channels,
                kernel_size=self.strides[0],
                stride=self.strides[0],
            )

        self.conv = ModuleList([conv])
        self.res = ModuleList([res])

        _channels = self._channels[0]
        for channels, size, stride in zip(
            self._channels[1:],
            self.kernel_sizes[1:],
            self.strides[1:],
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

    def forward(self, inputs, stream=None):
        """
        Parameters
        ----------
        inputs : Sequence[Tensor]
            shapes = [n, c * s, h, w] -- stream is None
                or
            shapes = [n, c, h, w] -- stream is int
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            shape = [n, c' * s, h // d, w // d] -- stream is None
                or
            shape = [n, c', h // d, w // d] -- stream is int
        """
        for conv, res in zip(self.conv, self.res):

            c = conv(inputs, stream=stream)
            r = res(inputs, stream=stream)

            inputs = [self.nonlinear(c) * self.gamma + r]

        return inputs[0]
