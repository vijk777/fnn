import torch
from itertools import chain

from .containers import Module


class Core(Module):
    def init(self, perspectives, grids, modulations):
        """
        Parameters
        ----------
        perspectives : int
            perspective channels
        grids : int
            grid channels
        modulations : int
            modulation channels
        """
        raise NotImplementedError

    @property
    def channels(self):
        raise NotImplementedError

    @property
    def grid_scale(self):
        raise NotImplementedError

    def forward(self, perspective, grid, modulation, dropout=0):
        """
        Parameters
        ----------
        perspective : Tensor
            shape = [n, c, h, w]
        grid : Tensor
            shape = [g, h, w]
        modulation : Tensor
            shape = [n, f]
        dropout : float
            dropout probability

        Returns
        -------
        Tensor
            shape = [n, c', h', w']
        """
        raise NotImplementedError()


class FeedforwardRecurrent(Core):
    def __init__(self, feedforward, recurrent):
        """
        Parameters
        ----------
        feedforward : .feedforwards.Feedforward
            feedforward network
        recurrent : .recurrents.Feedforward
            recurrent network
        """
        super().__init__()
        self.feedforward = feedforward
        self.recurrent = recurrent
        self.recurrent.add_input(self.feedforward.channels)

    def init(self, perspectives, grids, modulations):
        """
        Parameters
        ----------
        perspectives : int
            perspective channels
        grids : int
            grid channels
        modulations : int
            modulation channels
        """
        self.feedforward.add_input(perspectives)
        self.recurrent.add_input(grids)
        self.recurrent.add_input(modulations)

    @property
    def channels(self):
        return self.recurrent.channels

    @property
    def grid_scale(self):
        return self.feedforward.scale

    def forward(self, perspective, grid, modulation, dropout=0):
        """
        Parameters
        ----------
        perspective : Tensor
            shape = [n, c, h, w]
        grid : Tensor
            shape = [g, h, w]
        modulation : Tensor
            shape = [n, f]
        dropout : float
            dropout probability

        Returns
        -------
        Tensor
            shape = [n, c', h', w']
        """
        inputs = [
            self.feedforward([perspective]),
            grid[None, :, :, :],
            modulation[:, :, None, None],
        ]
        return self.recurrent(inputs, dropout=dropout)
