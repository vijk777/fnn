import torch
from torch import nn
from functools import reduce
from typing import Sequence, Optional

from .containers import Module, ModuleList
from .variables import EyePosition
from .elements import Linear, nonlinearity
from .utils import isotropic_grid_2d, rmat_3d


class Retina(Module):
    def __init__(self, eye_position: EyePosition):
        super().__init__()
        self.eye_position = eye_position

    def grid(self, height: int = 144, width: int = 256):
        """
        Args:
            height          (int)
            width           (int)
        Returns:
            (torch.Tensor)                  : shape = [h, w, 3]
        """
        raise NotImplementedError()

    def project(self, rmat: torch.Tensor, eye_position: Optional[torch.Tensor] = None):
        """
        Args:
            rmat            (torch.Tensor)  : shape = [n, 3, 3]
            eye_position    (torch.Tensor)  : shape = [n, f]
        Returns:
            (torch.Tensor)                  : shape = [n, h, w, 2]
        """
        raise NotImplementedError()

    def rays(self, rmat: torch.Tensor, height: int = 144, width: int = 256):
        """
        Args:
            rmat            (torch.Tensor)  : shape = [n, 3, 3]
            height          (int)
            width           (int)
        Returns:
            (torch.Tensor)                  : shape = [n, h, w, 3]
        """
        grid = self.grid(
            height=height,
            width=width,
        )
        return torch.einsum("N C D , H W D -> N H W C", rmat, grid)


class _MLP(Retina):
    def __init__(
        self,
        eye_position: EyePosition,
        features: Sequence[int],
        nonlinear: Optional[str] = None,
    ):
        super().__init__(eye_position=eye_position)

        self.features = list(map(int, features))
        self.layers = ModuleList()

        in_features = self.eye_position.n_features
        for out_features in self.features:

            linear = Linear(out_features=out_features).add(in_features=in_features)
            in_features = out_features

            self.layers.append(linear)

        self.proj = Linear(out_features=3).add(in_features=in_features)
        nn.init.constant_(self.proj.gain, 0)

        self.nonlinear, self.gamma = nonlinearity(nonlinear=nonlinear)

    def rmat(self, eye_position: Optional[torch.Tensor] = None):
        """
        Args:
            eye_position    (torch.Tensor)  : shape = [n, f]
        Returns:
            (torch.Tensor)                  : shape = [n, 3, 3]
        """
        x = self.eye_position(eye_position)
        x = reduce(lambda x, layer: self.nonlinear(layer([x])) * self.gamma, self.layers, x)
        x = self.proj([x])
        return rmat_3d(*x.unbind(1))


class AngularMLP(_MLP):
    def __init__(
        self,
        eye_position: EyePosition,
        degrees: float,
        features: Sequence[int],
        nonlinear: Optional[str] = None,
    ):
        super().__init__(
            eye_position=eye_position,
            features=features,
            nonlinear=nonlinear,
        )

        self.degrees = float(degrees)
        self.radians = self.degrees / 180 * torch.pi

    def grid(self, height: int = 144, width: int = 256):
        """
        Args:
            height          (int)
            width           (int)
        Returns:
            (torch.Tensor)                  : shape = [h, w, 3]
        """
        x, y = isotropic_grid_2d(
            height=height,
            width=width,
            device=self.device,
        ).unbind(2)

        r = torch.sqrt(x.pow(2) + y.pow(2)).mul(self.radians).clip(0, torch.pi / 2)
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)

        theta = torch.atan2(y, x)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        grid = [
            sin_r * cos_theta,
            sin_r * sin_theta,
            cos_r,
        ]
        return torch.stack(grid, dim=2)

    def rays(self, eye_position: Optional[torch.Tensor] = None, height: int = 144, width: int = 256):
        """
        Args:
            eye_position    (torch.Tensor)  : shape = [n, f]
            height          (int)
            width           (int)
        Returns:
            (torch.Tensor)                  : shape = [n, h, w, 3]
        """
        rmat = self.rmat(
            eye_position=eye_position,
        )
        grid = self.grid(
            height=height,
            width=width,
        )
        return torch.einsum("N C D , H W D -> N H W C", rmat, grid)

    def project(self, rays: torch.Tensor, eye_position: Optional[torch.Tensor] = None):
        """
        Args:
            rays            (torch.Tensor)  : shape = [n, h, w, 3]
            eye_position    (torch.Tensor)  : shape = [n, f]
        Returns:
            (torch.Tensor)                  : shape = [n, h, w, 2]
        """
        norm = rays.norm(p=2, dim=3, keepdim=True)
        rays = rays / norm

        rmat = self.rmat(eye_position=eye_position)

        x, y, z = torch.einsum("N H W C , N C D -> N H W D", rays, rmat).unbind(3)

        r = torch.arccos(z).div(self.radians)
        theta = torch.atan2(y, x)

        proj = [
            r * torch.cos(theta),
            r * torch.sin(theta),
        ]
        return torch.stack(proj, dim=3)

    def extra_repr(self):
        return f"degrees={self.degrees:.3g}"
