import torch
from itertools import chain

from .containers import Module
from .feedforwards import Feedforward
from .recurrents import Recurrent


class Core(Module):
    @property
    def out_channels(self):
        raise NotImplementedError

    @property
    def grid_scale(self):
        raise NotImplementedError

    def add_inputs(self, perspective, grid=None, modulation=None):
        """
        Parameters
        ----------
        perspective : int
            number of perspective channels
        grid : int | None
            number of perspective channels
        modulation : int | None
            number of modulation channels
        """
        raise NotImplementedError()

    def forward(self, perspective, grid, modulation, dropout=0):
        """
        Parameters
        ----------
        perspective : Sequence[Tensor]
            shapes = [n, c, h, w]
        grid : Sequence[Tensor | None]
            shapes = [n, c', h, w]
        modulation : Sequence[Tensor | None]
            shapes = [n, c', h, w]
        dropout : float
            dropout probability

        Returns
        -------
        Tensor
            shape = [n, c''', h, w]
        """
        raise NotImplementedError()


class FeedforwardRecurrent(Core):
    def __init__(self, feedforward: Feedforward, recurrent: Recurrent):
        super().__init__()

        recurrent.add_input(feedforward.out_channels)

        self.feedforward = feedforward
        self.recurrent = recurrent

    @property
    def out_channels(self):
        return self.recurrent.out_channels

    @property
    def grid_scale(self):
        return self.feedforward.scale

    def add_inputs(self, perspective, grid=None, modulation=None):
        """
        Parameters
        ----------
        perspective : int
            number of perspective channels
        grid : int | None
            number of perspective channels
        modulation : int | None
            number of modulation channels
        """
        self.feedforward.add_input(perspective)

        if grid is not None:
            self.recurrent.add_input(grid)

        if modulation is not None:
            self.recurrent.add_input(modulation)

    def forward(self, perspective, grid, modulation, dropout=0):
        """
        Parameters
        ----------
        perspective : Sequence[Tensor]
            shapes = [n, c, h, w]
        grid : Sequence[Tensor | None]
            shapes = [n, c', h, w]
        modulation : Sequence[Tensor | None]
            shapes = [n, c'', h, w]
        dropout : float
            dropout probability

        Returns
        -------
        Tensor
            shape = [n, c''', h', w']
        """
        inputs = (x for x in chain(*zip(grid, modulation)) if x is not None)
        inputs = [
            self.feedforward(perspective),
            *inputs,
        ]
        return self.recurrent(inputs=inputs, dropout=dropout)
