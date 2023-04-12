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
            feedforward channels per stream, F
        """
        raise NotImplementedError()

    @property
    def scale(self):
        """
        Returns
        -------
        int
            downscale factor, D
        """
        raise NotImplementedError()

    def init(self, inputs, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            [input channels per stream, I ...]
        streams : int
            number of streams, S
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
            feedforward channels per stream, F
        """
        return self._channels[-1]

    @property
    def scale(self):
        """
        Returns
        -------
        int
            downscale factor, D
        """
        return math.prod(self.strides)

    def init(self, inputs, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            [input channels per stream, I ...]
        streams : int
            number of streams, S
        """
        self.inputs = list(map(int, inputs))
        self.streams = int(streams)

        conv = Conv(channels=self._channels[0], streams=streams)
        res = Conv(channels=self._channels[0], streams=streams)

        for _channels in self.inputs:
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
        for conv, res in zip(self.conv, self.res):

            c = conv(inputs, stream=stream)
            r = res(inputs, stream=stream)

            inputs = [self.nonlinear(c) * self.gamma + r]

        return inputs[0]
