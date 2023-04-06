import torch

from .containers import Module


class Position(Module):
    def init(units):
        """
        Parameters
        ----------
        units : int
            number of units
        """
        raise NotImplementedError()
