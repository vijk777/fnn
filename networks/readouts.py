import torch

from .containers import Module


class Readout(Module):
    def init(self, cores, units):
        """
        Parameters
        ----------
        cores : int
            core channels, c
        units : int
            response units, u
        """
        raise NotImplementedError()

    def forward(self, core, dropout=0):
        """
        Parameters
        ----------
        core : Tensor
            shape = [n, c, h, w]
        dropout : float
            dropout probability

        Returns
        -------
        Tensor
            shape = [n, u]
        """
        raise NotImplementedError()
