import torch
from itertools import chain
from typing import Sequence

from .containers import Module
from .feedforwards import Feedforward
from .recurrents import Recurrent


class Core(Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = int(out_channels)

    def add_inputs(
        self,
        perspective: int,
        grid: int,
        modulation: int,
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
        grid: Sequence[torch.Tensor],
        modulation: Sequence[torch.Tensor],
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

        super().__init__(out_channels=recurrent.out_channels)

        self.feedforward = feedforward
        self.recurrent = recurrent
        self.recurrent.add_input(self.feedforward.out_channels)

    def add_inputs(
        self,
        perspective: int,
        grid: int,
        modulation: int,
    ):
        """
        Args:
            perspective (int)  : perspective channels
            grid        (int)  : grid channels
            modulation  (int)  : modulation channels
        """
        self.feedforward.add_input(perspective)
        self.recurrent.add_input(grid)
        self.recurrent.add_input(modulation)

    def forward(
        self,
        perspective: Sequence[torch.Tensor],
        grid: Sequence[torch.Tensor],
        modulation: Sequence[torch.Tensor],
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
        x = self.feedforward(perspective)
        x = self.recurrent([x, *chain(*zip(grid, modulation))], dropout=dropout)
        return x
