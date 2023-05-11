import torch
from .modules import Module


class Unit(Module):
    @property
    def readouts(self):
        """
        Returns
        -------
        int
            readouts per unit (R)
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


class Poisson(Unit):
    @property
    def readouts(self):
        return 1

    def forward(self, readout):
        return readout.squeeze(2).exp()
