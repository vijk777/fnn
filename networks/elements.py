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
    def __init__(self, drop_dim, reduce_dim):
        """
        Parameters
        ----------
        drop_dim : Sequence[int]
            dimensions to drop entirely
        reduce_dim : Sequence[int]
            dimensions to reduce across
        """
        super().__init__()
        self.drop_dim = list(map(int, sorted(drop_dim)))
        self.reduce_dim = list(map(int, sorted(reduce_dim)))
        self._past = dict()

    def _reset(self):
        self._past.clear()

    def forward(self, x, p=0):
        """
        Parameters
        ----------
        x : Tensor
            input
        p : float
            dropout probability

        Returns
        -------
        Tensor
            dropped input
        """
        if not self.training:
            return x

        if self._past:
            assert p == self._past.get("p")
            mask = self._past.get("mask")
        else:
            assert 0 <= p < 1
            if p:
                size = np.array(x.shape)
                size[self.drop_dim] = 1
                n = size[self.reduce_dim].prod().item()
                mask = torch.rand(size.tolist(), device=x.device) > p
                mask = n * mask / mask.sum(self.reduce_dim, keepdim=True).clip(1)
            else:
                mask = None
            self._past.update(p=p, mask=mask)

        if mask is None:
            return x
        else:
            return x.mul(mask)

    def extra_repr(self):
        return f"drop_dim={self.drop_dim}, reduce_dim={self.reduce_dim}"


class Input(Module):
    def __init__(
        self, in_channels, out_channels, groups=1, kernel_size=1, dynamic_size=1, stride=1, pad=True, mask=None
    ):
        """
        Parameters
        ----------
        in_channels : int
            input channels, must be divisible by groups
        out_channels : int
            output channels, must be divisible by groups
        groups : int
            number of groups
        kernel_size : int
            spatial kernel size
        dynamic_size : int
            temporal kernel size
        stride : int
            spatial stride
        pad : bool
            whether to pad to maintain spatial dimensions
        mask : Tensor (bool)
            masks the kernel, shape must be broadcastable with the kernel
        """
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")

        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        if (kernel_size - stride) % 2 != 0:
            raise ValueError("incompatible kernel_size and stride")

        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.kernel_size = int(kernel_size)
        self.dynamic_size = int(dynamic_size)
        self.stride = int(stride)
        self.pad = (self.kernel_size - self.stride) // 2 if pad else 0

        shape = [
            self.out_channels,
            self.in_channels // self.groups,
            self.dynamic_size,
            self.kernel_size,
            self.kernel_size,
        ]
        self.weight = nn.Parameter(torch.empty(shape))

        if mask is None:
            self.fan_in = math.prod(shape[1:])
            self.register_buffer("mask", None)
        else:
            assert mask.dtype == torch.bool
            mask = mask.expand_as(self.weight)

            self.fan_in = mask[0].sum().item()
            self.register_buffer("mask", mask)

        assert self.fan_in > 0
        bound = 1 / math.sqrt(self.fan_in)
        nn.init.uniform_(self.weight, -bound, bound)

        if self.mask is not None:
            with torch.no_grad():
                self.weight.mul_(mask)

        self._past = deque(maxlen=self.dynamic_size)

    def _reset(self):
        self._past.clear()

    def _param_norm_dims(self):
        yield self.weight, [1, 2, 3, 4]

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            shape = [n, c, h, w]

        Returns
        -------
        Tensor
            shape = [n, c, t, h, w]
        """
        if self.pad:
            x = F.pad(x, pad=[self.pad] * 4)

        x = x.unsqueeze(dim=2)

        if self.dynamic_size > 1:

            if not self._past:
                for _ in range(self.dynamic_size):
                    self._past.insert(0, torch.zeros_like(x))

            self._past.pop()
            self._past.insert(0, x)

            x = torch.cat([*self._past], dim=2)

        return x

    def extra_repr(self):
        s = (
            "{in_channels}, kernel_size=[{dynamic_size},{kernel_size},{kernel_size}]"
            ", groups={groups}, stride={stride}, pad={pad}"
        )
        return s.format(**self.__dict__)


class Conv(Module):
    def __init__(self, channels, groups=1, streams=1, gain=True, bias=True, eps=1e-5):
        """
        Parameters
        ----------
        channels : int
            output channels, must be divisible by (groups * streams)
        groups : int
            number of output groups per stream
        streams : int
            number of streams
        gain : bool
            output gain
        bias : bool
            output bias
        eps : float
            small value added to denominator for numerical stability
        """
        if channels % (groups * streams) != 0:
            raise ValueError("stream_channels must be divisible by stream_groups")

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

        self.stream_channels = self.channels // self.streams
        self.fan_out = self.stream_channels // self.groups
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
        """
        weight = self._weights.get(stream)

        if weight is not None:
            return weight

        weights, masks = zip(*((i[stream].weight, i[stream].mask) for i in self.inputs))

        _weights = (w if m is None else w[m] for w, m in zip(weights, masks))
        _weights = (w.view(self.stream_channels, -1) for w in _weights)
        _weights = torch.cat(list(_weights), dim=1)[:, :, None, None, None]

        var, mean = torch.var_mean(_weights, dim=1, unbiased=False, keepdim=True)
        fan_in = _weights.size(1)
        scale = (var * fan_in + self.eps).pow(-0.5)

        if self.use_gain:
            scale = scale * self.gains[stream].view_as(scale)

        weight = [(w - mean) * scale if m is None else m * (w - mean) * scale for w, m in zip(weights, masks)]
        self._weights[stream] = weight
        return weight

    def add_input(self, channels, groups=1, kernel_size=1, dynamic_size=1, stride=1, pad=True):
        """
        Parameters
        ----------
        channels : int
            input channels, must be divisible by (groups * streams)
        groups : int
            number of input groups per stream
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
        if channels % (groups * self.streams):
            raise ValueError("in_channels must be divisible by (in_groups * streams)")

        f = lambda: Input(
            in_channels=channels // self.streams,
            out_channels=self.stream_channels,
            groups=groups,
            kernel_size=kernel_size,
            dynamic_size=dynamic_size,
            stride=stride,
            pad=pad,
        )
        inputs = ModuleList([f() for _ in range(self.streams)])
        self.inputs.append(inputs)
        return self

    def add_intergroup(self):
        if self.groups <= 1:
            raise ValueError("there must be > 1 groups to add intergroup")

        g = self.groups
        c = self.stream_channels
        d = c // g

        m = ~torch.eye(g, device=self.device, dtype=torch.bool)
        m = m.view(g, 1, g, 1)
        m = m.expand(-1, d, -1, d)
        m = m.reshape(c, c, 1, 1, 1)

        f = lambda: Input(
            in_channels=c,
            out_channels=c,
            mask=m,
        )
        inputs = ModuleList([f() for _ in range(self.streams)])
        self.inputs.append(inputs)
        return self

    def forward(self, inputs, stream=None):
        """
        Parameters
        ----------
        inputs : Sequence[Tensor]
            shapes = [n, c, h, w] or broadcastable -- when stream is None
                or
            shapes = [n, c // streams, h, w] or broadcastable -- when stream is not None
        stream : int | None
            specific stream index (int) or all streams (None)

        Returns
        -------
        Tensor
            shape = [n, c', h, w] -- stream is None and pad==True
                or
            shape = [n, c', h', w'] -- stream is None and pad==False
                or
            shape = [n, c' // streams, h, w] -- stream is not None and pad==True
                or
            shape = [n, c' // streams, h', w'] -- stream is not None and pad==False
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

        for inps, x, weight in zip(self.inputs, inputs, weights):

            if stream is None:
                x = x.chunk(self.streams, dim=1)
                x = [inp(_x) for inp, _x in zip(inps, x)]
                x = torch.cat(x, dim=1)

                stride = inps[0].stride
                groups = inps[0].groups * self.streams

            else:
                x = inps[stream](x)

                stride = inps[stream].stride
                groups = inps[stream].groups

            out = F.conv3d(
                x,
                weight=weight,
                bias=bias,
                stride=stride,
                groups=groups,
            ).squeeze(dim=2)

            outs.append(out)
            bias = None

        return reduce(torch.Tensor.add_, outs)

    def extra_repr(self):
        s = "{channels}, groups={groups}, streams={streams}, gain={gain}, bias={bias}"
        return s.format(
            channels=self.channels,
            groups=self.groups,
            streams=self.streams,
            gain=self.use_gain,
            bias=self.use_bias,
        )


class Linear(Conv):
    def __init__(
        self,
        features,
        groups=1,
        streams=1,
        gain=True,
        bias=True,
        eps=1e-5,
    ):
        """
        Parameters
        ----------
        features : int
            output features, must be divisible by output groups
        groups : int
            number of output groups per stream
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
            input features, must be divisible by (groups * streams)
        groups : int
            number of input groups per stream
        """
        return super().add_input(channels=features, groups=groups)

    def forward(self, inputs, stream=None):
        """
        Parameters
        ----------
        inputs : Sequence[Tensor]
            shapes = [n, f] -- when stream is None
                or
            shapes = [n, f // streams] -- when stream is not None
        stream : int | None
            specific stream index (int) or all streams (None)

        Returns
        -------
        Tensor
            shape = [n, f'] -- when stream is None
                or
            shape = [n, f' // streams] -- when stream is not None
        """
        inputs = [x[:, :, None, None] for x in inputs]
        return super().forward(inputs, stream=stream)[:, :, 0, 0]
