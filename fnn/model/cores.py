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
            perspective channels per stream (P)
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
            [N, S*P, H, W] -- stream is None
                or
            [N, P, H, W] -- stream is int
        modulation : Tensor
            [N, S*M] -- stream is None
                or
            [N, M] -- stream is int
        stream : int | None
            specific stream (int) or all streams (None)

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
    """Feedforward & Recurrent Core"""

    def __init__(self, feedforward, recurrent):
        """
        Parameters
        ----------
        feedforward : fnn.model.feedforwards.Feedforward
            feedforward network
        recurrent : fnn.model.recurrents.Recurrent
            recurrent network
        """
        super().__init__()
        self.feedforward = feedforward
        self.recurrent = recurrent

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
            inputs=[self.perspectives],
            streams=self.streams,
        )
        self.recurrent._init(
            inputs=[self.feedforward.channels, self.modulations, 2],
            streams=self.streams,
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
        return self.recurrent.channels

    def forward(self, perspective, modulation, stream=None):
        """
        Parameters
        ----------
        perspective : Tensor
            [N, S*P, H, W] -- stream is None
                or
            [N, P, H, W] -- stream is int
        modulation : Tensor
            [N, S*M] -- stream is None
                or
            [N, M] -- stream is int
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, S*C, H', W'] -- stream is None
                or
            [N, C, H', W'] -- stream is int
        """
        f = self.feedforward([perspective], stream=stream)
        m = modulation[:, :, None, None]
        g = self._grid

        if g is None:
            _, _, H, W = f.shape
            g = isotropic_grid_2d(height=H, width=W, device=self.device)
            g = torch.einsum("H W C -> C H W", g)[None]
            self._grid = g

        if stream is None:
            g = g.repeat(1, self.streams, 1, 1)

        return self.recurrent([f, m, g], stream=stream)
