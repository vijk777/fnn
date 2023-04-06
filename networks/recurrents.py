import torch
from torch import nn
from typing import Sequence

from .containers import Module
from .elements import Dropout, Conv
from .utils import to_groups_2d


class Recurrent(Module):
    @property
    def out_channels(self):
        raise NotImplementedError

    def add_input(self, channels: int):
        raise NotImplementedError()

    def forward(self, inputs: Sequence[torch.Tensor], dropout: float = 0):
        """
        Args:
            inputs  (torch.Tensors) : shape = [n, c, h, w]
            dropout (float)         : dropout probability
        Returns:
            (torch.Tensor)          : shape = [n, c', h, w]
        """
        raise NotImplementedError()


class RvT(Recurrent):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        groups: int = 1,
    ):
        super().__init__()

        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.groups = int(groups)
        self.group_channels = self.channels // self.groups

        self.tau = nn.Parameter(torch.ones(self.groups))
        nn.init.constant_(self.tau, self.group_channels**-0.5)

        self.drop = Dropout(
            drop_dim=[4, 5],
            reduce_dim=[3],
        )

        self.proj_x = Conv(
            out_channels=self.channels,
            out_groups=self.groups,
            gain=False,
            bias=False,
        )
        if self.groups > 1:
            self.proj_x.add_intergroup()

        self.proj_q = Conv(out_channels=self.channels, out_groups=self.groups, gain=False, bias=False)
        self.proj_q.add(
            in_channels=self.channels * 2,
            in_groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.proj_k = Conv(out_channels=self.channels, out_groups=self.groups, gain=False, bias=False)
        self.proj_k.add(
            in_channels=self.channels * 2,
            in_groups=self.groups,
            kernel_size=self.kernel_size,
            pad=False,
        )

        self.proj_v = Conv(out_channels=self.channels, out_groups=self.groups, gain=False, bias=False)
        self.proj_v.add(
            in_channels=self.channels * 2,
            in_groups=self.groups,
            kernel_size=self.kernel_size,
            pad=False,
        )

        self.proj_i = Conv(out_channels=self.channels, out_groups=self.groups)
        self.proj_i.add(
            in_channels=self.channels,
            in_groups=self.groups,
        )
        self.proj_i.add(
            in_channels=self.channels * 2,
            in_groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.proj_f = Conv(out_channels=self.channels, out_groups=self.groups)
        self.proj_f.add(
            in_channels=self.channels,
            in_groups=self.groups,
        )
        self.proj_f.add(
            in_channels=self.channels * 2,
            in_groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.proj_g = Conv(out_channels=self.channels, out_groups=self.groups)
        self.proj_g.add(
            in_channels=self.channels,
            in_groups=self.groups,
        )
        self.proj_g.add(
            in_channels=self.channels * 2,
            in_groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.proj_o = Conv(out_channels=self.channels, out_groups=self.groups)
        self.proj_o.add(
            in_channels=self.channels,
            in_groups=self.groups,
        )
        self.proj_o.add(
            in_channels=self.channels * 2,
            in_groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self._past = dict()

    def _reset(self):
        self._past.clear()

    @property
    def out_channels(self):
        return self.channels

    def add_input(self, channels: int):
        self.proj_x.add(in_channels=channels)

    def forward(self, inputs: Sequence[torch.Tensor], dropout: float = 0):
        """
        Args:
            inputs (torch.Tensors)  : shape = [n, c, h, w]
            dropout (float)         : dropout probability
        Returns:
            (torch.Tensor)          : shape = [n, c', h, w]
        """
        if self._past:
            h = self._past["h"]
            c = self._past["c"]
        else:
            N, _, H, W = inputs[0].shape
            h = c = torch.zeros(N, self.out_channels, H, W, device=self.device)

        if self.groups > 1:
            x = self.proj_x([h, *inputs])
        else:
            x = self.proj_x(inputs)

        xh = [
            to_groups_2d(x, self.groups),
            to_groups_2d(h, self.groups),
        ]
        xh = torch.stack(xh, 2)
        xh = self.drop(xh, p=dropout).flatten(1, 3)

        q = to_groups_2d(self.proj_q([xh]), self.groups).flatten(3, 4)
        k = to_groups_2d(self.proj_k([xh]), self.groups).flatten(3, 4)
        v = to_groups_2d(self.proj_v([xh]), self.groups).flatten(3, 4)

        w = torch.einsum("G, N G C Q , N G C D -> N G Q D", self.tau, q, k).softmax(dim=3)
        a = torch.einsum("N G C D , N G Q D -> N G C Q", v, w).view_as(x)

        i = torch.sigmoid(self.proj_i([a, xh]))
        f = torch.sigmoid(self.proj_f([a, xh]))
        g = torch.tanh(self.proj_g([a, xh]))
        o = torch.sigmoid(self.proj_o([a, xh]))

        c = f * c + i * g
        h = o * torch.tanh(c)

        self._past["c"] = c
        self._past["h"] = h

        return h
