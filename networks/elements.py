import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
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
    def __init__(
        self,
        drop_dim,
        reduce_dim,
    ):
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
                z = size[self.reduce_dim].prod().item()

                mask = torch.rand(size.tolist(), device=x.device) > p
                z = z / mask.sum(self.reduce_dim, keepdim=True).clip(1)

                mask = mask * z
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
        self,
        in_channels,
        out_channels,
        groups=1,
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
            input channels, must be divisible by groups
        out_channels : int
            output channels, must be divisible by groups
        groups : int
            groups
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
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("channels must be divisible by groups")

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

        c = self.in_channels // self.groups
        shape = [self.out_channels, c, self.dynamic_size, self.kernel_size, self.kernel_size]

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

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Sequence[Tensor]
            shapes = [n, c, h, w]

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
    def __init__(
        self,
        out_channels,
        out_groups=1,
        gain=True,
        bias=True,
        eps=1e-5,
    ):
        """
        Parameters
        ----------
        out_channels : int
            output channels, must be divisible by output groups
        out_groups : int
            output groups
        gain : bool
            output gain
        bias : bool
            output bias
        eps : float
            small value added to denominator for numerical stability
        """
        if out_channels % out_groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        super().__init__()

        self.out_channels = int(out_channels)
        self.out_groups = int(out_groups)
        self.use_gain = bool(gain)
        self.use_bias = bool(bias)
        self.eps = float(eps)
        self.inputs = ModuleList()

        self.affine = []
        self.fan_out = self.out_channels // self.out_groups
        assert self.fan_out > 1

        if self.use_gain:
            self.gain = nn.Parameter(torch.ones(self.out_groups, self.fan_out))
            self.affine.append(self.gain)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.out_groups, self.fan_out))
            self.affine.append(self.bias)

        self._weights = None

    def _reset(self):
        self._weights = None

    def _param_norm_dims(self):
        for param in self.affine:
            yield param, 1

    def _param_groups(self, **kwargs):
        if self.affine and kwargs.get("weight_decay"):
            kwargs.update(weight_decay=0)
            yield dict(params=self.affine, **kwargs)

    @property
    def weights(self):
        if self._weights is None:

            masks = [i.mask for i in self.inputs]
            weights = [i.weight for i in self.inputs]

            m_weights = [weight if mask is None else weight[mask] for mask, weight in zip(masks, weights)]
            m_weights = [weight.view(self.out_channels, -1, 1, 1, 1) for weight in m_weights]
            m_weights = torch.cat(m_weights, dim=1)

            var, mean = torch.var_mean(m_weights, dim=1, unbiased=False, keepdim=True)
            scale = (var * m_weights.size(dim=1) + self.eps).pow(-0.5)

            if self.use_gain:
                scale = scale * self.gain.view_as(scale)

            scales = [scale if mask is None else scale * mask for mask in masks]

            self._weights = [(weight - mean) * scale for weight, scale in zip(weights, scales)]

        return self._weights

    @property
    def biases(self):
        biases = [None] * len(self.inputs)

        if self.use_bias:
            biases[-1] = self.bias.flatten()

        return biases

    def add(self, in_channels, in_groups=1, kernel_size=1, dynamic_size=1, stride=1, pad=True, mask=None):
        """
        Parameters
        ----------
        in_channels : int
            input channels
        in_groups : int
            input groups
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
        self.inputs.append(
            Input(
                in_channels=in_channels,
                out_channels=self.out_channels,
                groups=in_groups,
                kernel_size=kernel_size,
                dynamic_size=dynamic_size,
                stride=stride,
                pad=pad,
                mask=mask,
            )
        )
        return self

    def add_intergroup(self):
        if self.out_groups <= 1:
            raise ValueError("out_groups must be > 1")

        g = self.out_groups
        c = self.out_channels // g

        mask = ~torch.eye(g, device=self.device, dtype=torch.bool)
        mask = mask.view(g, 1, g, 1)
        mask = mask.expand(-1, c, -1, c)
        mask = mask.reshape(self.out_channels, self.out_channels, 1, 1, 1)

        return self.add(in_channels=self.out_channels, mask=mask)

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : Sequence[Tensor]
            shapes = [n, c, h, w]

        Returns
        -------
        Tensor
            shape = [n, c', h, w] (pad==True) or [n, c', h', w'] (pad==False)
        """
        weights = self.weights
        biases = self.biases
        assert len(inputs) == len(weights) == len(biases)
        outs = [
            F.conv3d(x=i(x), weight=w, bias=b, stride=i.stride, groups=i.groups).squeeze(dim=2)
            for i, x, w, b in zip(self.inputs, inputs, weights, biases)
        ]
        return reduce(torch.Tensor.add_, outs)

    def extra_repr(self):
        s = "{out_channels}, groups={out_groups}, gain={use_gain}, bias={use_bias}"
        return s.format(**self.__dict__)


class Linear(Conv):
    def __init__(
        self,
        out_features,
        out_groups=1,
        gain=True,
        bias=True,
        eps=1e-5,
    ):
        """
        Parameters
        ----------
        out_feature : int
            output feature, must be divisible by output groups
        out_groups : int
            output groups
        gain : bool
            output gain
        bias : bool
            output bias
        eps : float
            small value added to denominator for numerical stability
        """
        super().__init__(out_channels=out_features, out_groups=out_groups, gain=gain, bias=bias, eps=eps)

    def add(self, in_features, in_groups=1):
        """
        Parameters
        ----------
        in_features : int
            input features
        in_groups : int
            input groups
        """
        return super().add(in_channels=in_features, in_groups=in_groups)

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : Sequence[Tensor]
            shapes = [n, f]

        Returns
        -------
        Tensor
            shape = [n, f']
        """
        inputs = [x[:, :, None, None] for x in inputs]
        return super().forward(inputs)[:, :, 0, 0]
