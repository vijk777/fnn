import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from itertools import chain
from functools import reduce
from collections import deque

from .containers import Module, ModuleList


def nonlinearity(nonlinear=None):
    """
    Parameters
    ----------
    nonlinear : str | None
        "elu" | "silu" | "gelu" | "tanh" | None

    Adapted from: https://github.com/deepmind/deepmind-research/blob/cb555c241b20c661a3e46e5d1eb722a0a8b0e8f4/nfnets/base.py#L101
    """
    if nonlinear is None:
        return nn.Identity(), 1.0

    elif nonlinear == "elu":
        return nn.ELU(alpha=1.0), 1.2716004848480225

    elif nonlinear == "silu":
        return nn.SiLU(), 1.7881293296813965

    elif nonlinear == "gelu":
        return nn.GELU(), 1.7015043497085571

    elif nonlinear == "tanh":
        return nn.Tanh(), 1.5939117670059204

    else:
        raise NotImplementedError('"{}" not implemented'.format(nonlinear))


class Dropout(Module):
    def __init__(self, p=0):
        """
        Parameters
        ----------
        p : float
            dropout probability between 0 and 1
        """
        super().__init__()
        self.p = p

    def _reset(self):
        self.mask = None

    @property
    def p(self):
        if self.training:
            return self._p
        else:
            return 0

    @p.setter
    def p(self, p):
        assert 0 <= p < 1
        self._p = float(p)
        self.scale = 1 / (1 - self._p)
        self.mask = None

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            input

        Returns
        -------
        Tensor
            dropped input
        """
        if not self.p:
            return x

        if self.mask is None:
            N, C, _, _ = x.shape
            rand = torch.rand([N, C], device=x.device)
            self.mask = (rand > self.p) * self.scale

        return torch.einsum("N C H W , N C -> N C H W", x, self.mask)

    def extra_repr(self):
        return f"p={self.p:.3g}"


class Input(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        groups=1,
        streams=1,
        kernel_size=1,
        dynamic_size=1,
        stride=1,
        pad=True,
        mask=None,
    ):
        """
        Parameters
        ----------
        in_channels : int
            input channels per stream, must be divisible by groups
        out_channels : int
            output channels per stream, must be divisible by groups
        groups : int
            groups per stream
        kernel_size : int
            spatial kernel size
        dynamic_size : int
            temporal kernel size
        stride : int
            spatial stride
        pad : bool
            whether to pad to maintain spatial dimensions
        mask : Tensor | None
            dtype=torch.bool, shape must be broadcastable with the kernel
        """
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")

        if out_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")

        if (kernel_size - stride) % 2 != 0:
            raise ValueError("incompatible kernel_size and stride")

        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.streams = int(streams)
        self.kernel_size = int(kernel_size)
        self.dynamic_size = int(dynamic_size)
        self.past_size = self.dynamic_size - 1
        self.stride = int(stride)
        self.pad = (self.kernel_size - self.stride) // 2 if pad else 0

        shape = [
            self.out_channels,
            self.in_channels // self.groups,
            self.dynamic_size,
            self.kernel_size,
            self.kernel_size,
        ]

        if mask is None:
            self.fan_in = math.prod(shape[1:])
            self.register_buffer("mask", None)
        else:
            assert mask.dtype == torch.bool
            mask = mask.expand(shape)

            self.fan_in = mask.sum().item() // self.out_channels
            self.register_buffer("mask", mask)

        assert self.fan_in > 0
        bound = 1 / math.sqrt(self.fan_in)

        def param():
            weight = torch.zeros(shape)
            nn.init.uniform_(weight, -bound, bound)

            if self.mask is not None:
                weight.mul_(self.mask)

            return nn.Parameter(weight)

        self.weights = nn.ParameterList([param() for _ in range(self.streams)])

        self._past = [deque(maxlen=self.past_size) for _ in range(self.streams)]

    def _reset(self):
        for past in self._past:
            past.clear()

    def _param_norm_dims(self):
        for weight in self.weights:
            yield weight, [1, 2, 3, 4]

    def forward(self, x, stream):
        """
        Parameters
        ----------
        x : Tensor
            [N, C, H, W]
        stream : int
            stream index

        Returns
        -------
        Tensor
            [N, C, T, H, W]
        """
        if self.pad:
            x = F.pad(x, pad=[self.pad] * 4)

        if self.past_size:
            past = self._past[stream]
            if past:
                assert len(past) == self.past_size
            else:
                for _ in range(self.past_size):
                    past.insert(0, torch.zeros_like(x))

            out = torch.stack([x, *past], dim=2)
            past.pop()
            past.insert(0, x)
        else:
            out = x.unsqueeze(dim=2)

        return out

    def extra_repr(self):
        s = "({inp}->{out})x{streams}: groups={groups}, stride={stride}, pad={pad}"
        return s.format(
            inp=self.in_channels,
            out=self.out_channels,
            streams=self.streams,
            groups=self.groups,
            stride=self.stride,
            pad=self.pad,
        )


class Conv(Module):
    def __init__(self, channels, groups=1, streams=1, gain=True, bias=True, eps=1e-5):
        """
        Parameters
        ----------
        channels : int
            output channels per stream, must be divisible by groups
        groups : int
            groups per stream
        streams : int
            number of streams
        gain : bool
            output gain
        bias : bool
            output bias
        eps : float
            small value added to denominator for numerical stability
        """
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")

        super().__init__()

        self.channels = int(channels)
        self.groups = int(groups)
        self.streams = int(streams)
        self.use_gain = bool(gain)
        self.use_bias = bool(bias)
        self.eps = float(eps)

        self.inputs = ModuleList()
        self.gains = nn.ParameterList()
        self.biases = nn.ParameterList()

        self.fan_out = self.channels // self.groups
        assert self.fan_out > 1

        if self.use_gain:
            for _ in range(self.streams):
                self.gains.append(nn.Parameter(torch.ones(self.groups, self.fan_out)))

        if self.use_bias:
            for _ in range(self.streams):
                self.biases.append(nn.Parameter(torch.zeros(self.groups, self.fan_out)))

        self._weights = dict()

    def _reset(self):
        self._weights = dict()

    def _param_norm_dims(self):
        for param in chain(self.gains, self.biases):
            yield param, 1

    def _param_groups(self, **kwargs):
        params = list(chain(self.gains, self.biases))
        if params and kwargs.get("weight_decay"):
            kwargs.update(weight_decay=0)
            yield dict(params=params, **kwargs)

    def weights(self, stream):
        """
        Parameters
        ----------
        stream : int
            stream index

        Returns
        -------
        List[Tensor]
            [
                output channels,
                input channels // input groups,
                dynamic size,
                kernel size,
                kernel size
            ]
        """
        weights = self._weights.get(stream)

        if weights is not None:
            return weights

        weights, masks = zip(*((i.weights[stream], i.mask) for i in self.inputs))

        _weights = (w if m is None else w[m] for w, m in zip(weights, masks))
        _weights = (w.view(self.channels, -1) for w in _weights)
        _weights = torch.cat(list(_weights), dim=1)[:, :, None, None, None]

        var, mean = torch.var_mean(_weights, dim=1, unbiased=False, keepdim=True)
        fan_in = _weights.size(1)
        scale = (var * fan_in + self.eps).pow(-0.5)

        if self.use_gain:
            scale = scale * self.gains[stream].view_as(scale)

        weights = [(w - mean) * scale if m is None else m * (w - mean) * scale for w, m in zip(weights, masks)]

        self._weights[stream] = weights
        return weights

    def add_input(self, channels, groups=1, kernel_size=1, dynamic_size=1, stride=1, pad=True):
        """
        Parameters
        ----------
        channels : int
            input channels per stream, must be divisible by groups
        groups : int
            input groups per stream
        kernel_size : int
            spatial kernel size
        dynamic_size : int
            temporal kernel size
        stride : int
            spatial stride
        pad : bool
            spatial padding to preserve height and width
        mask : Tensor (bool)
            masks the kernel, shape must be broadcastable with the kernel
        """
        module = Input(
            in_channels=channels,
            out_channels=self.channels,
            groups=groups,
            streams=self.streams,
            kernel_size=kernel_size,
            dynamic_size=dynamic_size,
            stride=stride,
            pad=pad,
        )
        self.inputs.append(module)
        return self

    def add_intergroup(self):
        if self.groups <= 1:
            raise ValueError("there must be > 1 groups to add intergroup")

        g = self.groups
        c = self.channels
        d = c // g

        m = ~torch.eye(g, device=self.device, dtype=torch.bool)
        m = m.view(g, 1, g, 1)
        m = m.expand(-1, d, -1, d)
        m = m.reshape(c, c, 1, 1, 1)

        module = Input(
            in_channels=c,
            out_channels=c,
            mask=m,
            streams=self.streams,
        )
        self.inputs.append(module)
        return self

    def forward(self, inputs, stream=None):
        """
        Parameters
        ----------
        inputs : Sequence[Tensor]
            [N, S*C, H, W] -- stream is None
                or
            [N, C, H, W] -- stream is int
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, S*C', H, W] -- stream is None and pad==True
                or
            [N, S*C', H', W'] -- stream is None and pad==False
                or
            [N, C', H, W] -- stream is int and pad==True
                or
            [N, C', H', W'] -- stream is int and pad==False
        """
        assert len(inputs) == len(self.inputs)

        if stream is None:
            weights = (self.weights(s) for s in range(self.streams))
            weights = (torch.cat(w, dim=0) for w in zip(*weights))
        else:
            weights = self.weights(stream)

        if self.use_bias:
            if stream is None:
                bias = torch.cat([b.flatten() for b in self.biases])
            else:
                bias = self.biases[stream].flatten()
        else:
            bias = None

        outs = []

        for i, x, w in zip(self.inputs, inputs, weights):

            if stream is None:
                x = [i(_x, stream=s) for s, _x in enumerate(x.chunk(self.streams, dim=1))]
                x = torch.cat(x, dim=1)
                g = i.groups * self.streams
            else:
                x = i(x, stream=stream)
                g = i.groups

            out = F.conv3d(x, weight=w, groups=g, bias=bias, stride=i.stride).squeeze(dim=2)
            outs.append(out)
            bias = None

        return reduce(torch.Tensor.add_, outs)

    def extra_repr(self):
        s = "channels={channels}, groups={groups}, streams={streams}, gain={gain}, bias={bias}"
        return s.format(
            channels=self.channels,
            groups=self.groups,
            streams=self.streams,
            gain=self.use_gain,
            bias=self.use_bias,
        )


class Linear(Conv):
    def __init__(self, features, groups=1, streams=1, gain=True, bias=True, eps=1e-5):
        """
        Parameters
        ----------
        features : int
            features per stream, must be divisible by groups
        groups : int
            groups per stream
        streams : int
            number of streams
        gain : bool
            output gain
        bias : bool
            output bias
        eps : float
            small value added to denominator for numerical stability
        """
        super().__init__(channels=features, groups=groups, streams=streams, gain=gain, bias=bias, eps=eps)
        self.features = self.channels

    def add_input(self, features, groups=1):
        """
        Parameters
        ----------
        channels : int
            input features, must be divisible by groups
        groups : int
            number of input groups per stream
        """
        return super().add_input(channels=features, groups=groups)

    def forward(self, inputs, stream=None):
        """
        Parameters
        ----------
        inputs : Sequence[Tensor]
            [N, S*F] -- stream is None
                or
            [N, F] -- stream is int
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, S*F'] -- stream is None
                or
            [N, F'] -- stream is int
        """
        inputs = [x[:, :, None, None] for x in inputs]
        return super().forward(inputs, stream=stream)[:, :, 0, 0]
