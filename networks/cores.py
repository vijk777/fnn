import torch
from itertools import chain

from .containers import Module


class Core(Module):
    def __init__(
        self,
        perspectives,
        grids,
        modulations,
    ):
        """
        Parameters
        ----------
        perspectives : int
            perspective channels
        grids : int
            perspective channels
        modulations : int
            modulation channels
        """
        super().__init__()
        self.perspectives = int(perspectives)
        self.grids = int(grids)
        self.modulations = int(modulations)

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
    def __init__(
        self,
        perspectives,
        grids,
        modulations,
        feedforward,
        recurrent,
    ):
        """
        Parameters
        ----------
        perspectives : int
            perspective channels
        grids : int
            perspective channels
        modulations : int
            modulation channels
        feedforward : .feedforwards.Feedforward
            feedforward network
        recurrent : .recurrents.Feedforward
            recurrent network
        """
        super().__init__(
            perspectives=perspectives,
            grids=grids,
            modulations=modulations,
        )

        self.feedforward = feedforward
        self.feedforward.add_input(self.perspectives)

        self.recurrent = recurrent
        self.recurrent.add_input(self.feedforward.channels)
        self.recurrent.add_input(self.grids)
        self.recurrent.add_input(self.modulations)

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
