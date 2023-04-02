import torch
from typing import Sequence, Optional

from .containers import Module, ModuleList
from .elements import Conv, nonlinearity


class Feedforward(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)


class Res3d(Feedforward):
    def __init__(
        self,
        in_channels: int,
        out_channels: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        nonlinear: Optional[str] = None,
    ):
        assert len(out_channels) == len(kernel_sizes) == len(strides)
        super().__init__(in_channels, out_channels[-1])

        self.conv = ModuleList()
        self.res = ModuleList()

        in_channels = self.in_channels
        for out_channels, size, stride in zip(out_channels, kernel_sizes, strides):

            conv = Conv(out_channels=out_channels).add(
                in_channels=in_channels,
                kernel_size=size,
                dynamic_size=size,
                stride=stride,
            )
            self.conv.append(conv)

            res = Conv(out_channels=out_channels).add(
                in_channels=in_channels,
                kernel_size=stride,
                stride=stride,
            )
            torch.nn.init.constant_(res.gain, 0)
            self.res.append(res)

            in_channels = out_channels

        self.nonlinear, self.gamma = nonlinearity(nonlinear)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): shape = [n, c, h, w]
        Returns:
            (torch.Tensor): shape = [n, c', h', w']
        """
        for conv, res in zip(self.conv, self.res):
            c = conv([x])
            r = res([x])
            x = self.nonlinear(c).mul(self.gamma).add(r)

        return x
