import torch
from numpy.typing import ArrayLike
from typing import Optional

from .containers import Module


def default_stimulus(n_channels: int = 1):
    return Stimulus(
        mean=[0] * n_channels,
        std=[1] * n_channels,
    )


def default_eye_position(n_features: int = 2):
    return EyePosition(
        mean=[0] * n_features,
        std=[1] * n_features,
    )


def default_behavior(n_features: int = 3):
    return Behavior(
        mean=[0] * n_features,
        std=[1] * n_features,
        zero=[False] * n_features,
    )


def default_response(n_units: int = 100):
    return Response(
        mean=[1] * n_units,
    )


class Stimulus(Module):
    def __init__(
        self,
        mean: ArrayLike,
        std: ArrayLike,
        eps: float = 1e-5,
    ):
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

        self.n_channels = self.mean.numel()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor)    : shape = [n, c, h, w]
        Returns:
            (torch.Tensor)      : shape = [n, c, h, w]
        """
        x = x.to(device=self.device)

        mean = self.mean[:, None, None]
        std = self.std[:, None, None]

        return (x - mean) / std

    def extra_repr(self):
        f = ", ".join(["{:.3g}"] * self.n_channels)
        f = f"mean=[{f}], std=[{f}]"
        return f.format(*self.mean.tolist(), *self.std.tolist())


class EyePosition(Module):
    def __init__(
        self,
        mean: ArrayLike,
        std: ArrayLike,
        eps: float = 1e-5,
    ):
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

        self.n_features = self.mean.numel()

    def forward(self, x: Optional[torch.Tensor] = None):
        """
        Args:
            x (torch.Tensor)    : shape = [n, f]
        Returns:
            (torch.Tensor)      : shape = [n, f]
        """
        if x is None:
            return torch.zeros(1, self.n_features, device=self.device)
        else:
            x = x.to(device=self.device)
            return (x - self.mean) / self.std

    def extra_repr(self):
        f = ", ".join(["{:.3g}"] * self.n_features)
        f = f"mean=[{f}], std=[{f}]"
        return f.format(*self.mean.tolist(), *self.std.tolist())


class Behavior(Module):
    def __init__(
        self,
        mean: ArrayLike,
        std: ArrayLike,
        zero: ArrayLike,
        eps: float = 1e-5,
    ):
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
        self.n_features = len(self.mean)

        assert self.mean.ndim == self.std.ndim == self.zero.ndim == 1
        assert self.mean.numel() == self.std.numel() == self.zero.numel()
        assert self.std.min().item() > eps

        self.n_features = self.mean.numel()

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
            return torch.zeros(1, self.n_features, device=self.device)
        else:
            x = x.to(device=self.device)
            return (x - self.adj_mean) / self.std

    def extra_repr(self):
        f = ", ".join(["{:.3g}"] * self.n_features)
        m = ", ".join(["{}"] * self.n_features)
        f = f"mean=[{f}], std=[{f}], zero=[{m}]"
        return f.format(*self.mean.tolist(), *self.std.tolist(), *self.zero.tolist())


class Response(Module):
    def __init__(
        self,
        mean: ArrayLike,
        eps: float = 1e-5,
    ):
        """
        Args:
            mean    (ArrayLike) : shape = [number of units], dtype = float
        """
        super().__init__()

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float))

        assert self.mean.ndim == 1
        assert self.mean.min().item() > eps

        self.n_units = self.mean.numel()

    def forward(
        self,
        x: torch.Tensor,
        log: bool = False,
        inverse: bool = False,
    ):
        """
        Args:
            x       (torch.Tensor)  : shape = [n, u]
            log     (bool)          : log response or normal response
            inverse (bool)          : raw or standardize
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
        return f"n_units={self.n_units}"
