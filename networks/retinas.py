import torch

from .containers import Module, ModuleList
from .utils import isotropic_grid_2d


class Retina(Module):
    def grid(self, height=144, width=256):
        """
        Parameters
        ----------
        height : int
            output height, h
        width : int
            output width, w

        Returns
        -------
        Tensor
            shape = [h, w, 3]
        """
        raise NotImplementedError()

    def project(self, rays, rmat):
        """
        Parameters
        ----------
        rays : Tensor
            shape = [n, h, w, 3]
        rmat : Tensor
            shape = [n, 3, 3], orthonormal 3D rotation matrix

        Returns
        -------
        Tensor
            shape = [n, h, w, 2]
        """
        raise NotImplementedError()

    def rays(self, rmat, height=144, width=256):
        """
        Parameters
        ----------
        rmat : Tensor
            shape = [n, 3, 3], orthonormal 3D rotation matrix
        height : int
            output height, h
        width : int
            output width, w

        Returns
        -------
        Tensor
            shape = [n, h, w, 3]
        """
        grid = self.grid(
            height=height,
            width=width,
        )
        return torch.einsum("N C D , H W D -> N H W C", rmat, grid)


class Angular(Retina):
    def __init__(
        self,
        degrees,
    ):
        """
        Parameters
        ----------
        degrees : float
            maximum visual degrees
        """
        super().__init__()
        self.degrees = float(degrees)
        self.radians = self.degrees / 180 * torch.pi

    def grid(self, height=144, width=256):
        """
        Parameters
        ----------
        height : int
            output height, h
        width : int
            output width, w

        Returns
        -------
        Tensor
            shape = [h, w, 3]
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

    def project(self, rays, rmat):
        """
        Parameters
        ----------
        rays : Tensor
            shape = [n, h, w, 3]
        rmat : Tensor
            shape = [n, 3, 3], orthonormal 3D rotation matrix

        Returns
        -------
        Tensor
            shape = [n, h, w, 2]
        """
        norm = rays.norm(p=2, dim=3, keepdim=True)
        rays = rays / norm

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
