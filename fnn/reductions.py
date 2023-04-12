import torch

from .containers import Module


class Reduce(Module):
    def init(self, dim, keepdim=False):
        """
        Parameters
        ----------
        dim : Sequence[int]
            dimensions to reduce
        keepdim : bool
            whether the output tensor has dim retained or not
        """
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()


class Mean(Reduce):
    def init(self, dim, keepdim=False):
        """
        Parameters
        ----------
        dim : Sequence[int]
            dimensions to reduce
        keepdim : bool
            whether the output tensor has dim retained or not
        """
        self.dim = list(map(int, dim))
        self.keepdim = bool(keepdim)

    def forward(self, x):
        return x.mean(dim=self.dim, keepdim=self.keepdim)
