import torch
import numpy as np
from typing import Sequence, Optional

from .containers import Module, ModuleList
from .elements import Conv, nonlinearity


class Feedforward(Module):
    def __init__(self, out_channels: int, downscale: int = 1):
        super().__init__()
        self.out_channels = int(out_channels)
        self.downscale = int(downscale)

    def add_input(self, channels: int):
        raise NotImplementedError()

    def forward(self, inputs: Sequence[torch.Tensor]):
        """
        Args:
            inputs (torch.Tensors)  : shape = [n, c, h, w]
        Returns:
            (torch.Tensor)          : shape = [n, c', h, w]
        """
        raise NotImplementedError()


class Res3d(Feedforward):
    def __init__(
        self,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        nonlinear: Optional[str] = None,
    ):
        self.channels = list(map(int, channels))
        self.kernel_sizes = list(map(int, kernel_sizes))
        self.strides = list(map(int, strides))

        assert len(self.channels) == len(self.kernel_sizes) == len(self.strides)

        super().__init__(
            out_channels=self.channels[-1],
            downscale=np.prod(self.strides),
        )

        self.conv = ModuleList([Conv(out_channels=self.channels[0])])
        self.res = ModuleList([Conv(out_channels=self.channels[0])])

        in_channels = self.channels[0]
        for channels, size, stride in zip(
            self.channels[1:],
            self.kernel_sizes[1:],
            self.strides[1:],
        ):
            conv = Conv(out_channels=channels).add(
                in_channels=in_channels,
                kernel_size=size,
                dynamic_size=size,
                stride=stride,
            )
            res = Conv(out_channels=channels).add(
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

    def add_input(self, channels: int):
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

    def forward(self, inputs: Sequence[torch.Tensor]):
        """
        Args:
            inputs (Sequence of torch.Tensors)  : shape = [n, c, h, w]
        Returns:
            (torch.Tensor)                      : shape = [n, c', h, w]
        """
        for conv, res in zip(self.conv, self.res):
            c = conv(inputs)
            r = res(inputs)
            inputs = [self.nonlinear(c) * self.gamma + r]

        return inputs[0]
