import torch
from .modules import Module
from .elements import Conv
from .utils import isotropic_grid_2d


# -------------- Core Prototype --------------


class Core(Module):
    """Core Module"""

    def _init(self, perspectives, modulations, streams):
        """
        Parameters
        ----------
        perspectives : int
            perspective channels (P)
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

    def forward(self, perspective, modulation, stream=None):
        """
        Parameters
        ----------
        perspective : Tensor
            [N, P, H, W]
        modulation : Tensor
            [N, M]
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
        recurrent : fnn.model.recurrents.Recurrent
            recurrent network
        channels : int
            output channels per stream
        """
        super().__init__()
        self.feedforward = feedforward
        self.recurrent = recurrent
        self._channels = int(channels)

    def _init(self, perspectives, modulations, streams):
        """
        Parameters
        ----------
        perspectives : int
            perspective channels (P)
        modulations : int
            modulation features per stream (M)
        streams : int
            number of streams (S)
        """
        self.perspectives = int(perspectives)
        self.modulations = int(modulations)
        self.streams = int(streams)
        self.feedforward._init(
            inputs=[
                [self.perspectives, False],
            ],
            streams=self.streams,
        )
        self.recurrent._init(
            inputs=[
                [self.feedforward.channels, True],
                [self.modulations, True],
                [2, False],
            ],
            streams=self.streams,
        )
        self.proj = Conv(channels=self.channels, streams=self.streams, gain=False, bias=False)
        self.proj.add_input(
            channels=self.recurrent.channels,
            drop=True,
        )
        self._reset()

    def _reset(self):
        self._grid = None

    @property
    def channels(self):
        """
        Returns
        -------
        int
            core channels per stream (C)
        """
        return self._channels

    def forward(self, perspective, modulation, stream=None):
        """
        Parameters
        ----------
        perspective : Tensor
            [N, P, H, W]
        modulation : Tensor
            [N, M]
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
            modulation = modulation.repeat(1, self.streams)

        N = max(perspective.size(0), modulation.size(0))
        f = self.feedforward([perspective], stream=stream).expand(N, -1, -1, -1)
        m = modulation[:, :, None, None].expand(N, -1, -1, -1)
        g = self._grid

        if g is None:
            _, _, H, W = f.shape
            g = isotropic_grid_2d(height=H, width=W, device=self.device)
            g = torch.einsum("H W C -> C H W", g)[None]
            self._grid = g

        if stream is None:
            g = g.repeat(1, self.streams, 1, 1)

        r = self.recurrent([f, m, g], stream=stream)

        return self.proj([r], stream=stream)
