import torch
from torch import nn

from .containers import Module
from .elements import Dropout, Conv
from .utils import to_groups_2d


class Recurrent(Module):
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
            downscale factor, d
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
            shape = [n, c' * s, h // d, w // d] -- stream is None
                or
            shape = [n, c', h // d, w // d] -- stream is int
        """
        raise NotImplementedError()


class RvT(Recurrent):
    def __init__(self, channels, groups, kernel_size):
        """
        Parameters
        ----------
        channels : int
            recurrent channels per stream
        groups : int
            groups per stream
        kernel_size : int
            kernel size
        """
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")

        super().__init__()

        self._channels = int(channels)
        self.groups = int(groups)
        self.kernel_size = int(kernel_size)
        self.drop = Dropout()
        self._past = dict()

    def _reset(self):
        self._past.clear()

    @property
    def channels(self):
        """
        Returns
        -------
        int
            output channels per stream, c'
        """
        return self._channels

    @property
    def scale(self):
        """
        Returns
        -------
        int
            downscale factor, d
        """
        return 1

    def init(self, channels, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            input channels per stream, c
        streams : int
            number of streams, s
        """
        self.streams = int(streams)

        self.proj_x = Conv(channels=self.channels, groups=self.groups, streams=streams, gain=False, bias=False)
        for c in channels:
            self.proj_x.add_input(channels=c)
        if self.groups > 1:
            self.proj_x.add_intergroup()

        init = (self.channels / self.groups) ** -0.5
        tau = lambda: nn.Parameter(torch.full([self.groups], init))
        self.tau = nn.ParameterList([tau() for _ in range(streams)])

        self.proj_q = Conv(channels=self.channels, groups=self.groups, streams=streams, gain=False, bias=False)
        self.proj_q.add_input(
            channels=self.channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.proj_k = Conv(channels=self.channels, groups=self.groups, streams=streams, gain=False, bias=False)
        self.proj_k.add_input(
            channels=self.channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
            pad=False,
        )

        self.proj_v = Conv(channels=self.channels, groups=self.groups, streams=streams, gain=False, bias=False)
        self.proj_v.add_input(
            channels=self.channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
            pad=False,
        )

        self.proj_i = Conv(channels=self.channels, groups=self.groups, streams=streams)
        self.proj_i.add_input(
            channels=self.channels,
            groups=self.groups,
        )
        self.proj_i.add_input(
            channels=self.channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.proj_f = Conv(channels=self.channels, groups=self.groups, streams=streams)
        self.proj_f.add_input(
            channels=self.channels,
            groups=self.groups,
        )
        self.proj_f.add_input(
            channels=self.channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.proj_g = Conv(channels=self.channels, groups=self.groups, streams=streams)
        self.proj_g.add_input(
            channels=self.channels,
            groups=self.groups,
        )
        self.proj_g.add_input(
            channels=self.channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
        )

        self.proj_o = Conv(channels=self.channels, groups=self.groups, streams=streams)
        self.proj_o.add_input(
            channels=self.channels,
            groups=self.groups,
        )
        self.proj_o.add_input(
            channels=self.channels * 2,
            groups=self.groups,
            kernel_size=self.kernel_size,
        )

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
        if stream is None:
            channels = self.channels * self.streams
            groups = self.groups * self.streams
            tau = torch.cat(list(self.tau))
        else:
            channels = self.channels
            groups = self.groups
            tau = self.tau[stream]

        if self._past:
            assert self._past["stream"] == stream
            h = self._past["h"]
            c = self._past["c"]
        else:
            self._past["stream"] = stream
            N, _, H, W = inputs[0].shape
            h = c = torch.zeros(N, channels, H, W, device=self.device)

        if self.groups > 1:
            x = self.proj_x([*inputs, h], stream=stream)
        else:
            x = self.proj_x(inputs, stream=stream)

        xh = [
            to_groups_2d(x, groups),
            to_groups_2d(h, groups),
        ]
        xh = torch.stack(xh, 2).flatten(1, 3)
        xh = self.drop(xh)

        q = to_groups_2d(self.proj_q([xh], stream=stream), groups).flatten(3, 4)
        k = to_groups_2d(self.proj_k([xh], stream=stream), groups).flatten(3, 4)
        v = to_groups_2d(self.proj_v([xh], stream=stream), groups).flatten(3, 4)

        w = torch.einsum("G , N G C Q , N G C D -> N G Q D", tau, q, k).softmax(dim=3)
        a = torch.einsum("N G C D , N G Q D -> N G C Q", v, w).view_as(x)

        i = torch.sigmoid(self.proj_i([a, xh], stream=stream))
        f = torch.sigmoid(self.proj_f([a, xh], stream=stream))
        g = torch.tanh(self.proj_g([a, xh], stream=stream))
        o = torch.sigmoid(self.proj_o([a, xh], stream=stream))

        c = f * c + i * g
        h = o * torch.tanh(c)

        self._past["c"] = c
        self._past["h"] = h

        return h
