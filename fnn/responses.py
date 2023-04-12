import torch

from .modules import Module


class Response(Module):
    @property
    def readouts(self):
        """
        Returns
        -------
        int
            readouts (R) per unit (U)
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
