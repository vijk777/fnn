import torch
from torch import nn

from .containers import Module


class Response(Module):
    @property
    def parameters(self):
        """
        Returns
        -------
        int
            response parameters (R)
        """
        raise NotImplementedError()

    def forward(self, readout):
        """
        Parameters
        ----------
        readout : Tensor
            [N, U, R]

        Returns
        -------
        Tensor
            [N, U]
        """
        raise NotImplementedError()

    def loss(self, readout, target):
        """
        Parameters
        ----------
        readout : Tensor
            [N, U, R]
        target : Tensor
            [N, U]

        Returns
        -------
        Tensor
            [N, U]
        """
        raise NotImplementedError()
