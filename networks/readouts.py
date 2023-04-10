import torch
from torch import nn

from .containers import Module
from .elements import Dropout, Conv


class Readout(Module):
    def init(self, units, cores, streams):
        """
        Parameters
        ----------
        units : int
            number of units, u
        cores : int
            core channels per stream, c
        streams : int
            number of streams, s
        """
        raise NotImplementedError()

    def forward(self, core, stream=None):
        """
        Parameters
        ----------
        core : Tensor
            shape = [n, s * c, h, w] -- stream is None
                or
            shape = [n, c, h, w] -- stream is int
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            shape = [n, u]
        """
        raise NotImplementedError()


class PositionFeatures(Readout):
    def __init__(self, channels, position, bound, features):
        """
        Parameters
        ----------
        channels : int
            readout channels
        position : .positions.Position
            spatial position
        bounds : .bounds.Bound
            spatial bound
        features : .features.Features
            feature weights
        """
        assert bound.vmin == -1 and bound.vmax == 1
        super().__init__()

        self.channels = int(channels)
        self.position = position
        self.bound = bound
        self.features = features
        self.drop = Dropout()

    def init(self, cores, units):
        """
        Parameters
        ----------
        cores : int
            core channels, c
        units : int
            response units, u
        """
        self.proj.add_input(
            in_channels=cores,
        )
        self.position.init(
            units=units,
        )
        self.features.init(
            units=units,
            features=self.channels,
        )
        self.bias = nn.Parameter(
            torch.zeros(units),
        )

    def _param_groups(self, **kwargs):
        if kwargs.get("weight_decay"):
            kwargs.update(weight_decay=0)
            yield dict(params=[self.bias], **kwargs)
