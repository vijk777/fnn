from .modules import Module


class Core(Module):
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

    def init(self, perspectives, grids, modulations, streams):
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

    def forward(self, perspective, grid, modulation, stream=None):
        """
        Parameters
        ----------
        perspective : Tensor
            [N, P, H, W]
        grid : Tensor
            [G, H, W]
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
            core channels per stream (C)
        """
        return self.recurrent.channels

    @property
    def grid_scale(self):
        """
        Returns
        -------
        int
            grid downscale factor (D)
        """
        return self.feedforward.scale

    def init(self, perspectives, grids, modulations, streams):
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

        self.feedforward.init(
            inputs=[
                [perspectives, False],
            ],
            streams=streams,
        )
        self.recurrent.init(
            inputs=[
                [self.feedforward.channels, True],
                [grids, False],
                [modulations, True],
            ],
            streams=streams,
        )

    def forward(self, perspective, grid, modulation, stream=None):
        """
        Parameters
        ----------
        perspective : Tensor
            [N, P, H, W]
        grid : Tensor
            [G, H, W]
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

        return self.recurrent(inputs, stream=stream)
