import torch
from numpy.typing import ArrayLike
from typing import Optional

from .containers import Module


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
        self.n_channels = len(self.mean)

        assert self.n_channels == self.mean.numel() == self.std.numel()
        assert self.std.min().item() > eps

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor)    : shape = [N, C, H, W]
        Returns:
            (torch.Tensor)      : shape = [N, C, H, W]
        """
        x = x.to(device=self.device)

        mean = self.mean[:, None, None]
        std = self.std[:, None, None]

        return (x - mean) / std


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
        self.n_features = len(self.mean)

        assert self.n_features == self.mean.numel() == self.std.numel()
        assert self.std.min().item() > eps

    def forward(self, x: Optional[torch.Tensor] = None):
        """
        Args:
            x (torch.Tensor)    : shape = [N, C, H, W]
        Returns:
            (torch.Tensor)      : shape = [N, C, H, W]
        """
        if x is None:
            return torch.zeros(1, self.n_features, device=self.device)
        else:
            x = x.to(device=self.device)
            return (x - self.mean) / self.std


class Behavior(Module):
    def __init__(
        self,
        mean: ArrayLike,
        std: ArrayLike,
        mask: ArrayLike,
        eps: float = 1e-5,
    ):
        """
        Args:
            mean        (ArrayLike) : shape = [number of features], dtype = float
            std         (ArrayLike) : shape = [number of features], dtype = float
            mask        (ArrayLike) : shape = [number of features], dtype = bool
        """
        super().__init__()

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float))
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.bool))
        self.n_features = len(self.mean)

        assert self.n_features == self.mean.numel() == self.std.numel() == self.mask.numel()
        assert self.std.min().item() > eps

        adj_mean = self.mean.clone()
        adj_mean[self.mask] = 0
        self.register_buffer("adj_mean", adj_mean)

    def forward(self, x: Optional[torch.Tensor] = None):
        """
        Args:
            x (torch.Tensor)    : shape = [N, F]
        Returns:
            (torch.Tensor)      : shape = [N, F]
        """
        if x is None:
            return torch.zeros(1, self.n_features, device=self.device)
        else:
            x = x.to(device=self.device)
            return (x - self.adj_mean) / self.std


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
        self.n_units = len(self.mean)

        assert self.n_units == self.mean.numel()
        assert self.mean.min().item() > eps

    def forward(
        self,
        x: torch.Tensor,
        log: bool = False,
        inverse: bool = False,
    ):
        """
        Args:
            x       (torch.Tensor)  : shape = [N, U]
            log     (bool)          : log response or normal response
            inverse (bool)          : raw or standardize
        Returns:
            (torch.Tensor)          : shape = [N, U]
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
