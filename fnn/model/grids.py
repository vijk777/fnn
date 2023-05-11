import torch
from .modules import Module


# -------------- Grid Prototype --------------


class Grid(Module):
    """Grid Module"""

    @property
    def grids(self):
        """
        Returns
        -------
        int
            grid channels (G)
        """
        raise NotImplementedError()

    def forward(self, grid):
        """
        Parameters
        ----------
        grid : Tensor
            [H, W, 3], grid of 3D points on a unit sphere

        Returns
        -------
        Tensor
            [G, H, W]
        """
        raise NotImplementedError()


# -------------- Grid Types --------------


class Raw(Grid):
    @property
    def grids(self):
        """
        Returns
        -------
        int
            grid channels (G)
        """
        return 3

    def forward(self, grid):
        """
        Parameters
        ----------
        grid : Tensor
            [H, W, 3], grid of 3D points on a unit sphere

        Returns
        -------
        Tensor
            [G, H, W]
        """
        return torch.einsum("H W G -> G H W", grid)
