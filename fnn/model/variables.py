import torch
from numpy.typing import ArrayLike
from typing import Optional

from .modules import Module


def default_stimulus(channels: int = 1):
    return Stimulus(
        mean=[0] * channels,
        std=[1] * channels,
    )


def default_eye_position(features: int = 2):
    return EyePosition(
        mean=[0] * features,
        std=[1] * features,
    )


def default_behavior(features: int = 3):
    return Behavior(
        mean=[0] * features,
        std=[1] * features,
        zero=[False] * features,
    )


def default_response(units: int = 100):
    return Response(
        mean=[1] * units,
    )


class Stimulus(Module):
    def __init__(self, mean: ArrayLike, std: ArrayLike, eps: float = 1e-5):
        """
        Args:
            mean    (ArrayLike) : shape = [number of channels], dtype = float
            std     (ArrayLike) : shape = [number of channels], dtype = float
        """
        super().__init__()

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float))

        assert self.mean.ndim == self.std.ndim == 1
        assert self.mean.numel() == self.std.numel()
        assert self.std.min().item() > eps

        self.channels = self.mean.numel()

    def forward(self, x: torch.Tensor, inverse: bool = False):
        """
        Args:
            x       (torch.Tensor)  : shape = [n, c, h, w]
            inverse (bool)          : inverse transform or normal transform
        Returns:
            (torch.Tensor)      : shape = [n, c, h, w]
        """
        x = x.to(device=self.device)
        mean = self.mean[:, None, None]
        std = self.std[:, None, None]

        if inverse:
            return x * std + mean
        else:
            return (x - mean) / std

    def extra_repr(self):
        f = ", ".join(["{:.3g}"] * self.channels)
        f = f"mean=[{f}], std=[{f}]"
        return f.format(*self.mean.tolist(), *self.std.tolist())


class EyePosition(Module):
    def __init__(self, mean: ArrayLike, std: ArrayLike, eps: float = 1e-5):
        """
        Args:
            mean    (ArrayLike) : shape = [number of features], dtype = float
            std     (ArrayLike) : shape = [number of features], dtype = float
        """
        super().__init__()

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float))

        assert self.mean.ndim == self.std.ndim == 1
        assert self.mean.numel() == self.std.numel()
        assert self.std.min().item() > eps

        self.features = self.mean.numel()

    def forward(self, x: Optional[torch.Tensor] = None):
        """
        Args:
            x (torch.Tensor)    : shape = [n, f]
        Returns:
            (torch.Tensor)      : shape = [n, f]
        """
        if x is None:
            return torch.zeros(1, self.features, device=self.device)
        else:
            x = x.to(device=self.device)
            return (x - self.mean) / self.std

    def extra_repr(self):
        f = ", ".join(["{:.3g}"] * self.features)
        f = f"mean=[{f}], std=[{f}]"
        return f.format(*self.mean.tolist(), *self.std.tolist())


class Behavior(Module):
    def __init__(self, mean: ArrayLike, std: ArrayLike, zero: ArrayLike, eps: float = 1e-5):
        """
        Args:
            mean        (ArrayLike) : shape = [number of features], dtype = float
            std         (ArrayLike) : shape = [number of features], dtype = float
            zero        (ArrayLike) : shape = [number of features], dtype = bool
        """
        super().__init__()

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float))
        self.register_buffer("zero", torch.tensor(zero, dtype=torch.bool))
        self.features = len(self.mean)

        assert self.mean.ndim == self.std.ndim == self.zero.ndim == 1
        assert self.mean.numel() == self.std.numel() == self.zero.numel()
        assert self.std.min().item() > eps

        self.features = self.mean.numel()

        adj_mean = self.mean.clone()
        adj_mean[self.zero] = 0
        self.register_buffer("adj_mean", adj_mean)

    def forward(self, x: Optional[torch.Tensor] = None):
        """
        Args:
            x (torch.Tensor)    : shape = [n, f]
        Returns:
            (torch.Tensor)      : shape = [n, f]
        """
        if x is None:
            return torch.zeros(1, self.features, device=self.device)
        else:
            x = x.to(device=self.device)
            return (x - self.adj_mean) / self.std

    def extra_repr(self):
        f = ", ".join(["{:.3g}"] * self.features)
        m = ", ".join(["{}"] * self.features)
        f = f"mean=[{f}], std=[{f}], zero=[{m}]"
        return f.format(*self.mean.tolist(), *self.std.tolist(), *self.zero.tolist())


class Response(Module):
    def __init__(self, mean: ArrayLike, eps: float = 1e-5):
        """
        Args:
            mean    (ArrayLike) : shape = [number of units], dtype = float
        """
        super().__init__()

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float))

        assert self.mean.ndim == 1
        assert self.mean.min().item() > eps

        self.units = self.mean.numel()

    def forward(self, x: torch.Tensor, log: bool = False, inverse: bool = False):
        """
        Args:
            x       (torch.Tensor)  : shape = [n, u]
            log     (bool)          : log response or normal response
            inverse (bool)          : inverse transform or normal transform
        Returns:
            (torch.Tensor)          : shape = [n, u]
        """
        x = x.to(device=self.device)

        if log:
            if inverse:
                return x + self.mean.log()
            else:
                return x - self.mean.log()
        else:
            if inverse:
                return x * self.mean
            else:
                return x / self.mean

    def extra_repr(self):
        return f"units={self.units}"
