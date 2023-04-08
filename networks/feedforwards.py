import torch
import math

from .containers import Module, ModuleList
from .elements import Conv, nonlinearity


class Feedforward(Module):
    @property
    def channels(self):
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
    def __init__(
        self,
        channels,
        kernel_sizes,
        strides,
        nonlinear=None,
    ):
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
        super().__init__()

        self._channels = list(map(int, channels))
        self.kernel_sizes = list(map(int, kernel_sizes))
        self.strides = list(map(int, strides))

        assert len(self._channels) == len(self.kernel_sizes) == len(self.strides)

        self.conv = ModuleList([Conv(out_channels=self._channels[0])])
        self.res = ModuleList([Conv(out_channels=self._channels[0])])

        in_channels = self._channels[0]
        for channels, size, stride in zip(
            self._channels[1:],
            self.kernel_sizes[1:],
            self.strides[1:],
        ):
            conv = Conv(out_channels=channels).add_input(
                in_channels=in_channels,
                kernel_size=size,
                dynamic_size=size,
                stride=stride,
            )
            res = Conv(out_channels=channels).add_input(
                in_channels=in_channels,
                kernel_size=stride,
                stride=stride,
            )
            in_channels = channels
            self.conv.append(conv)
            self.res.append(res)

        for res in self.res:
            torch.nn.init.constant_(res.gain, 0)

        self.nonlinear, self.gamma = nonlinearity(nonlinear)

    @property
    def channels(self):
        return self._channels[-1]

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
        self.conv[0].add(
            in_channels=channels,
            kernel_size=self.kernel_sizes[0],
            dynamic_size=self.kernel_sizes[0],
            stride=self.strides[0],
        )
        self.res[0].add(
            in_channels=channels,
            kernel_size=self.strides[0],
            stride=self.strides[0],
        )

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
        for conv, res in zip(self.conv, self.res):
            c = conv(inputs)
            r = res(inputs)
            inputs = [self.nonlinear(c) * self.gamma + r]

        return inputs[0]
