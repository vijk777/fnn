import torch
from torch import nn

from .containers import Module


class Reduce(Module):
    def init(self, streams):
        """
        Parameters
        ----------
        streams : int
            number of streams (S)
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
    def init(self, streams):
        """
        Parameters
        ----------
        streams : int
            number of streams (S)
        """
        self.streams = int(streams)

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
        return x.mean(dim=1)
