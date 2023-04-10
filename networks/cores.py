import torch
from itertools import chain

from .containers import Module


class Core(Module):
    @property
    def channels(self):
        """
        Returns
        -------
        int
            output channels per stream, c'
        """
        raise NotImplementedError()

    @property
    def grid_scale(self):
        """
        Returns
        -------
        int
            grid downscale factor, d
        """
        raise NotImplementedError()

    def init(self, perspectives, grids, modulations, streams):
        """
        Parameters
        ----------
        perspectives : int
            perspective channels, c
        grids : int
            grid channels, g
        modulations : int
            modulation features per stream, m
        streams : int
            number of streams, s
        """
        raise NotImplementedError()

    def forward(self, perspective, grid, modulation, stream=None):
        """
        Parameters
        ----------
        perspective : Tensor
            shape = [n, c, h, w]
        grid : Tensor
            shape = [g, h, w]
        modulation : Tensor
            shape = [n, s*m] -- stream is None
                or
            shape = [n, m] -- stream is int
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            shape = [n, s*c', h', w'] -- stream is None
                or
            shape = [n, c', h', w'] -- stream is int
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

    @property
    def channels(self):
        """
        Returns
        -------
        int
            output channels per stream, c'
        """
        return self.recurrent.channels

    @property
    def grid_scale(self):
        """
        Returns
        -------
        int
            grid downscale factor, d
        """
        return self.feedforward.scale

    def init(self, perspectives, grids, modulations, streams):
        """
        Parameters
        ----------
        perspectives : int
            perspective channels, c
        grids : int
            grid channels, g
        modulations : int
            modulation features per stream, m
        streams : int
            number of streams, s
        """
        self.streams = int(streams)
        self.feedforward.init(
            channels=[perspectives],
            streams=streams,
        )
        self.recurrent.init(
            channels=[self.feedforward.channels, grids, modulations],
            streams=streams,
        )

    def forward(self, perspective, grid, modulation, stream=None):
        """
        Parameters
        ----------
        perspective : Tensor
            shape = [n, c, h, w]
        grid : Tensor
            shape = [g, h, w]
        modulation : Tensor
            shape = [n, m * s] -- stream is None
                or
            shape = [n, m] -- stream is int
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            shape = [n, s*c', h', w'] -- stream is None
                or
            shape = [n, c', h', w'] -- stream is int
        """
        if stream is None:
            perspective = perspective.repeat(1, self.streams, 1, 1)
            grid = grid.repeat(self.streams, 1, 1)

        inputs = [
            self.feedforward([perspective], stream=stream),
            grid[None, :, :, :],
            modulation[:, :, None, None],
        ]

        return self.recurrent(inputs, stream=stream)
