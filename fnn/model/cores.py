import torch
from .modules import Module


# -------------- Core Base --------------


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
            masks=[True],
            streams=self.streams,
        )
        self.recurrent._init(
            inputs=[self.feedforward.channels, self.modulations],
            masks=[True, False],
            streams=self.streams,
        )

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
        r = self.recurrent([f, m], stream=stream)
        return r
