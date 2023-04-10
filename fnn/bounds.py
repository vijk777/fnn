import torch

from .containers import Module


class Bound(Module):
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



class Tanh(Bound):
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
