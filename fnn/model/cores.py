from .modules import Module
from .elements import Conv


# -------------- Core Prototype --------------


class Core(Module):
    """Core Module"""

    def _init(self, perspectives, grids, modulations, streams):
        """
        Parameters
        ----------
        perspectives : int
            perspective channels (P)
        grids : int
            grid channels (G)
        modulations : int
            modulation features per stream (M)
        streams : int
            number of streams (S)
        """
        raise NotImplementedError()

    @property
    def channels(self):
        """
        Returns
        -------
        int
            core channels per stream (C)
        """
        raise NotImplementedError()

    @property
    def grid_scale(self):
        """
        Returns
        -------
        int
            grid downscale factor (D)
        """
        raise NotImplementedError()

    def forward(self, perspective, grid, modulation, stream=None):
        """
        Parameters
        ----------
        perspective : Tensor
            [N, P, H, W]
        grid : Tensor
            [G, H/D, W/D]
        modulation : Tensor
            [N, S*M] -- stream is None
                or
            [N, M] -- stream is int
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            [N, S*C, H', W'] -- stream is None
                or
            [N, C, H', W'] -- stream is int
        """
        raise NotImplementedError()


# -------------- Core Types --------------


class FeedforwardRecurrent(Core):
    def __init__(self, feedforward, recurrent, channels):
        """
        Parameters
        ----------
        feedforward : fnn.model.feedforwards.Feedforward
            feedforward network
        recurrent : fnn.model.recurrents.Feedforward
            recurrent network
        channels : int
            output channels per stream
        """
        super().__init__()
        self.feedforward = feedforward
        self.recurrent = recurrent
        self._channels = int(channels)

    @property
    def channels(self):
        """
        Returns
        -------
        int
            core channels per stream (C)
        """
        return self._channels

    @property
    def grid_scale(self):
        """
        Returns
        -------
        int
            grid downscale factor (D)
        """
        return self.feedforward.scale

    def _init(self, perspectives, grids, modulations, streams):
        """
        Parameters
        ----------
        perspectives : int
            perspective channels (P)
        grids : int
            grid channels (G)
        modulations : int
            modulation features per stream (M)
        streams : int
            number of streams (S)
        """
        self.perspectives = int(perspectives)
        self.grids = int(grids)
        self.modulations = int(modulations)
        self.streams = int(streams)

        self.feedforward._init(
            inputs=[
                [perspectives, False],
            ],
            streams=streams,
        )
        self.recurrent._init(
            inputs=[
                [self.feedforward.channels, True],
                [grids, False],
                [modulations, True],
            ],
            streams=streams,
        )

        self.proj = Conv(channels=self.channels, streams=self.streams, gain=False, bias=False)
        self.proj.add_input(
            channels=self.recurrent.channels,
            drop=True,
        )

    def forward(self, perspective, grid, modulation, stream=None):
        """
        Parameters
        ----------
        perspective : Tensor
            [N, P, H, W]
        grid : Tensor
            [G, H/D, W/D]
        modulation : Tensor
            [N, S*M] -- stream is None
                or
            [N, M] -- stream is int
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            [N, S*C, H', W'] -- stream is None
                or
            [N, C, H', W'] -- stream is int
        """
        if stream is None:
            perspective = perspective.repeat(1, self.streams, 1, 1)
            grid = grid.repeat(self.streams, 1, 1)

        inputs = [
            self.feedforward([perspective], stream=stream),
            grid[None, :, :, :],
            modulation[:, :, None, None],
        ]
        out = self.recurrent(inputs, stream=stream)

        return self.proj([out], stream=stream)
