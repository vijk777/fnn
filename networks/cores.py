import torch
from itertools import chain
from typing import Optional, Sequence

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

    def add_inputs(
        self,
        perspective: int,
        grid: Optional[int] = None,
        modulation: Optional[int] = None,
    ):
        """
        Args:
            perspective (int)  : perspective channels
            grid        (int)  : grid channels
            modulation  (int)  : modulation channels
        """
        raise NotImplementedError()

    def forward(
        self,
        perspective: Sequence[torch.Tensor],
        grid: Sequence[Optional[torch.Tensor]],
        modulation: Sequence[Optional[torch.Tensor]],
        dropout: float = 0,
    ):
        """
        Args:
            perspective (torch.Tensors) : shape = [n, c, h, w]
            grid        (torch.Tensors) : shape = [n, c, h, w]
            modulation  (torch.Tensors) : shape = [n, c, h, w]
            dropout     (float)         : dropout probability
        Returns:
            (torch.Tensor)              : shape = [n, c', h, w]
        """
        raise NotImplementedError()


class FeedforwardRecurrent(Core):
    def __init__(self, feedforward: Feedforward, recurrent: Recurrent):
        super().__init__()
        self.feedforward = feedforward
        self.recurrent = recurrent
        self.recurrent.add_input(self.feedforward.out_channels)

    @property
    def out_channels(self):
        return self.recurrent.out_channels

    @property
    def grid_scale(self):
        return self.feedforward.scale

    def add_inputs(
        self,
        perspective: int,
        grid: Optional[int] = None,
        modulation: Optional[int] = None,
    ):
        """
        Args:
            perspective (int)  : perspective channels
            grid        (int)  : grid channels
            modulation  (int)  : modulation channels
        """
        self.feedforward.add_input(perspective)

        if grid is not None:
            self.recurrent.add_input(grid)

        if modulation is not None:
            self.recurrent.add_input(modulation)

    def forward(
        self,
        perspective: Sequence[torch.Tensor],
        grid: Sequence[Optional[torch.Tensor]],
        modulation: Sequence[Optional[torch.Tensor]],
        dropout: float = 0,
    ):
        """
        Args:
            perspective (torch.Tensors) : shape = [n, c, h, w]
            grid        (torch.Tensors) : shape = [n, c, h, w]
            modulation  (torch.Tensors) : shape = [n, c, h, w]
            dropout     (float)         : dropout probability
        Returns:
            (torch.Tensor)              : shape = [n, c', h, w]
        """
        inputs = (x for x in chain(*zip(grid, modulation)) if x is not None)
        inputs = [
            self.feedforward(perspective),
            *inputs,
        ]
        return self.recurrent(inputs=inputs, dropout=dropout)
