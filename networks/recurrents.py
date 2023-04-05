import torch
from torch import nn
from typing import Sequence

from .containers import Module
from .elements import Dropout, Conv
from .utils import to_groups_2d


class Recurrent(Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
    ):
        super().__init__()
        self.in_channels = list(map(int, in_channels))
        self.out_channels = int(out_channels)


class RvT(Recurrent):
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
    ):
        super().__init__(in_channels, out_channels)

        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.groups = int(groups)
        self.group_channels = self.out_channels // self.groups

        self.tau = nn.Parameter(torch.ones(self.groups))
        nn.init.constant_(self.tau, self.group_channels**-0.5)

        self.drop = Dropout(
            drop_dim=[4, 5],
            reduce_dim=[3],
        )

        self.proj_x = Conv(
            out_channels=self.out_channels,
            out_groups=self.groups,
            gain=False,
            bias=False,
        )

        for c in self.in_channels:
            self.proj_x.add(in_channels=c)

        if self.groups > 1:
            self.proj_x.add_intergroup()

        self.proj_q = Conv(out_channels=self.out_channels, out_groups=self.groups, gain=False, bias=False)
        self.proj_q.add(
            in_channels=self.out_channels * 2,
            in_groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.proj_k = Conv(out_channels=self.out_channels, out_groups=self.groups, gain=False, bias=False)
        self.proj_k.add(
            in_channels=self.out_channels * 2,
            in_groups=self.groups,
            kernel_size=self.kernel_size,
            pad=False,
        )

        self.proj_v = Conv(out_channels=self.out_channels, out_groups=self.groups, gain=False, bias=False)
        self.proj_v.add(
            in_channels=self.out_channels * 2,
            in_groups=self.groups,
            kernel_size=self.kernel_size,
            pad=False,
        )

        self.proj_i = Conv(out_channels=self.out_channels, out_groups=self.groups)
        self.proj_i.add(
            in_channels=self.out_channels,
            in_groups=self.groups,
        )
        self.proj_i.add(
            in_channels=self.out_channels * 2,
            in_groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.proj_f = Conv(out_channels=self.out_channels, out_groups=self.groups)
        self.proj_f.add(
            in_channels=self.out_channels,
            in_groups=self.groups,
        )
        self.proj_f.add(
            in_channels=self.out_channels * 2,
            in_groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.proj_g = Conv(out_channels=self.out_channels, out_groups=self.groups)
        self.proj_g.add(
            in_channels=self.out_channels,
            in_groups=self.groups,
        )
        self.proj_g.add(
            in_channels=self.out_channels * 2,
            in_groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.proj_o = Conv(out_channels=self.out_channels, out_groups=self.groups)
        self.proj_o.add(
            in_channels=self.out_channels,
            in_groups=self.groups,
        )
        self.proj_o.add(
            in_channels=self.out_channels * 2,
            in_groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self._past = dict()

    def _reset(self):
        self._past.clear()

    def forward(
        self,
        inputs: Sequence[torch.Tensor],
        dropout: float = 0,
    ):
        """
        Args:
            inputs (Sequence of torch.Tensors): shape = [n, c, h, w]
            dropout (float): dropout probability
        Returns:
            (torch.Tensor): shape = [n, c', h, w]
        """
        if self._past:
            h = self._past["h"]
            c = self._past["c"]
        else:
            h = c = torch.zeros(1, self.out_channels, 1, 1, device=self.device)

        if self.groups > 1:
            x = self.proj_x([*inputs, h])
        else:
            x = self.proj_x(inputs)

        xh = torch.stack([to_groups_2d(x, self.groups), to_groups_2d(h.expand_as(x), self.groups)], 2)
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
