import torch
from .modules import Module

# -------------- Bound Prototype --------------


class Bound(Module):
    """Bound Module"""

    @property
    def vmin(self):
        raise NotImplementedError()

    @property
    def vmax(self):
        raise NotImplementedError()

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor

        Returns
        -------
        Tensor
            x bounded between vmin and vmax
        """
        raise NotImplementedError()


# -------------- Bound Types --------------


class Tanh(Bound):
    """Tanh Bound"""

    @property
    def vmin(self):
        return -1

    @property
    def vmax(self):
        return 1

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor

        Returns
        -------
        Tensor
            x bounded between -1, and 1
        """
        return torch.tanh(x)
