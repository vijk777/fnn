import torch
from itertools import chain

from .containers import Module


class Core(Module):
    def init(self, perspectives, grids, modulations):
        """
        Parameters
        ----------
        perspectives : int
            perspective channels, c
        grids : int
            grid channels, g
        modulations : int
            modulation features, m
        """
        raise NotImplementedError()

    def forward(self, perspective, grid, modulation, dropout=0):
        """
        Parameters
        ----------
        perspective : Tensor
            shape = [n, c, h, w] -- stream is None
                or
            shape = [n, c // s, h, w] -- stream is int
        grid : Tensor
            shape = [g, h, w] -- stream is None
                or
            shape = [g // s, h, w] -- stream is int
        modulation : Tensor
            shape = [n, m] -- stream is None
                or
            shape = [n, m // s] -- stream is int
        stream : int | None
            specific stream (int) or all streams (None)
        dropout : float
            dropout probability

        Returns
        -------
        Tensor
            shape = [n, c', h', w'] -- stream is None
                or
            shape = [n, c' // s, h', w'] -- stream is None
        """
        raise NotImplementedError()

    @property
    def channels(self):
        raise NotImplementedError()

    @property
    def streams(self):
        raise NotImplementedError()

    @property
    def grid_scale(self):
        raise NotImplementedError


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
        assert feedforward.streams == recurrent.streams
        super().__init__()

        self.feedforward = feedforward
        self.recurrent = recurrent

        self.recurrent.add_input(channels=self.feedforward.channels)

    def init(self, perspectives, grids, modulations):
        """
        Parameters
        ----------
        perspectives : int
            perspective channels, c
        grids : int
            grid channels, g
        modulations : int
            modulation features, m
        """
        self.feedforward.add_input(
            channels=perspectives,
        )
        self.recurrent.add_input(
            channels=grids,
        )
        self.recurrent.add_input(
            channels=modulations,
        )

    def forward(self, perspective, grid, modulation, stream=None, dropout=0):
        """
        Parameters
        ----------
        perspective : Tensor
            shape = [n, c, h, w] -- stream is None
                or
            shape = [n, c // s, h, w] -- stream is int
        grid : Tensor
            shape = [g, h, w] -- stream is None
                or
            shape = [g // s, h, w] -- stream is int
        modulation : Tensor
            shape = [n, m] -- stream is None
                or
            shape = [n, m // s] -- stream is int
        stream : int | None
            specific stream (int) or all streams (None)
        dropout : float
            dropout probability

        Returns
        -------
        Tensor
            shape = [n, c', h', w'] -- stream is None
                or
            shape = [n, c' // s, h', w'] -- stream is None
        """
        inputs = [
            self.feedforward([perspective], stream=stream),
            grid[None, :, :, :],
            modulation[:, :, None, None],
        ]
        return self.recurrent(inputs, stream=stream, dropout=dropout)

    @property
    def channels(self):
        return self.recurrent.channels

    @property
    def streams(self):
        return self.recurrent.streams

    @property
    def grid_scale(self):
        return self.feedforward.scale
