import torch
from itertools import chain

from .containers import Module


class Core(Module):
    def __init__(
        self,
        perspective_channels,
        grid_channels,
        modulation_channels,
    ):
        """
        Parameters
        ----------
        perspective_channels : int
            perspective channels
        grid_channels : int
            perspective channels
        modulation_channels : int
            modulation channels
        """
        super().__init__()
        self.perspective_channels = int(perspective_channels)
        self.grid_channels = int(grid_channels)
        self.modulation_channels = int(modulation_channels)

    @property
    def out_channels(self):
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
            shape = [n, c', h, w]
        modulation : Tensor
            shape = [n, f]
        dropout : float
            dropout probability

        Returns
        -------
        Tensor
            shape = [n, c'', h', w']
        """
        raise NotImplementedError()


class FeedforwardRecurrent(Core):
    def __init__(
        self,
        perspective_channels,
        grid_channels,
        modulation_channels,
        feedforward,
        recurrent,
    ):
        """
        Parameters
        ----------
        perspective_channels : int
            perspective channels
        grid_channels : int
            perspective channels
        modulation_channels : int
            modulation channels
        feedforward : .feedforwards.Feedforward
            feedforward network
        recurrent : .recurrents.Feedforward
            recurrent network
        """
        super().__init__(
            perspective_channels=perspective_channels,
            grid_channels=grid_channels,
            modulation_channels=modulation_channels,
        )

        self.feedforward = feedforward
        self.feedforward.add_input(self.perspective_channels)

        self.recurrent = recurrent
        self.recurrent.add_input(self.feedforward.out_channels)
        self.recurrent.add_input(self.grid_channels)
        self.recurrent.add_input(self.modulation_channels)

    @property
    def out_channels(self):
        return self.recurrent.out_channels

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
            shape = [n, c', h, w]
        modulation : Tensor
            shape = [n, f]
        dropout : float
            dropout probability

        Returns
        -------
        Tensor
            shape = [n, c'', h', w']
        """
        inputs = [
            self.feedforward([perspective]),
            grid,
            modulation[:, :, None, None],
        ]
        return self.recurrent(inputs, dropout=dropout)
