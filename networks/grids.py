import torch

from .containers import Module


class Grid(Module):
    @property
    def grids(self):
        raise NotImplementedError()

    def forward(self, grid):
        """
        Parameters
        ----------
        grid : Tensor
            shape = [h, w, 3]

        Returns
        -------
        Tensor
            shape = [g, h, w] or [g, h, w]
        """
        raise NotImplementedError()


class Vanilla(Grid):
    @property
    def grids(self):
        return 3

    def forward(self, grid):
        """
        Parameters
        ----------
        grid : Tensor
            shape = [h, w, 3]

        Returns
        -------
        Tensor
            shape = [g, h, w] or [g, h, w]
        """
        return torch.einsum("H W G -> G H W", grid)
