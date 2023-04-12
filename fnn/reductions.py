import torch

from .containers import Module


class Reduce(Module):
    def init(self, dim):
        """
        Parameters
        ----------
        dim : Sequence[int]
            dimensions to reduce
        """
        raise NotImplementedError()

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            [N, S, U, R]

        Returns
        -------
        Tensor
            [N, U, R]
        """
        raise NotImplementedError()


class Mean(Reduce):
    def init(self, dim):
        """
        Parameters
        ----------
        dim : Sequence[int]
            dimensions to reduce
        """
        self.dim = list(map(int, dim))

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            [N, S, U, R]

        Returns
        -------
        Tensor
            [N, U, R]
        """
        return x.mean(dim=self.dim)
