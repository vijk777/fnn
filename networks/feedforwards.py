import torch
import numpy as np
from typing import Sequence, Optional

from .containers import Module, ModuleList
from .standardization import Stimulus
from .elements import Conv, nonlinearity


class Feedforward(Module):
    def __init__(self, stimulus: Stimulus, out_channels: int, downscale: int):
        super().__init__()
        self.stimulus = stimulus
        self.out_channels = int(out_channels)
        self.downscale = int(downscale)

    def forward(self, stimulus: torch.Tensor):
        """
        Args:
            stimulus (torch.Tensor) : shape = [n, c, h, w]
        Returns:
            (torch.Tensor)          : shape = [n, c', h', w']
        """
        raise NotImplementedError()


class Res3d(Feedforward):
    def __init__(
        self,
        stimulus: Stimulus,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        nonlinear: Optional[str] = None,
    ):
        self.channels = list(map(int, channels))
        self.kernel_sizes = list(map(int, kernel_sizes))
        self.strides = list(map(int, strides))

        assert len(self.channels) == len(self.kernel_sizes) == len(self.strides)

        super().__init__(stimulus, self.channels[-1], np.prod(self.strides))

        self.conv = ModuleList()
        self.res = ModuleList()

        in_channels = self.stimulus.n_channels
        for out_channels, size, stride in zip(self.channels, self.kernel_sizes, self.strides):

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

    def forward(self, stimulus: torch.Tensor):
        """
        Args:
            stimulus (torch.Tensor) : shape = [n, c, h, w]
        Returns:
            (torch.Tensor)          : shape = [n, c', h', w']
        """
        x = self.stimulus(stimulus)

        for conv, res in zip(self.conv, self.res):
            c = conv([x])
            r = res([x])
            x = self.nonlinear(c).mul(self.gamma).add(r)

        return x
