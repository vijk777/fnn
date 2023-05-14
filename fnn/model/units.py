import torch
from .modules import Module


# -------------- Unit Prototype --------------


class Unit(Module):
    """Unit Module"""

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

    def loss(self, readout, unit):
        """
        Parameters
        ----------
        readout : Tensor
            [N, U, R]
        unit : Tensor
            [N, U]

        Returns
        -------
        Tensor
            [N, U]
        """
        raise NotImplementedError()


# -------------- Unit Types --------------


class Poisson(Unit):
    @property
    def readouts(self):
        """
        Returns
        -------
        int
            readouts per unit (R)
        """
        return 1

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
        return readout.squeeze(2).exp()

    def loss(self, readout, unit):
        """
        Parameters
        ----------
        readout : Tensor
            [N, U, R]
        unit : Tensor
            [N, U]

        Returns
        -------
        Tensor
            [N, U]
        """
        r = readout.squeeze(2)
        return r.exp() - r * unit
