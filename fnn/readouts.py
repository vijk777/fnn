import torch
from torch import nn

from .containers import Module
from .elements import Dropout, Conv


class Readout(Module):
    def init(self, cores, outputs, units, streams):
        """
        Parameters
        ----------
        cores : int
            core channels per stream (C)
        outputs : int
            outputs per unit and stream (O)
        units : int
            number of units (U)
        streams : int
            number of streams (S)
        """
        raise NotImplementedError()

    def forward(self, core, stream=None):
        """
        Parameters
        ----------
        core : Tensor
            [N, S*C, H, W] -- stream is None
                or
            [N, C, H, W] -- stream is int
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            [N, S, U, O] -- stream is None
                or
            [N, S, O] -- stream is int
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

    def init(self, cores, outputs, units, streams):
        """
        Parameters
        ----------
        cores : int
            core channels per stream (C)
        outputs : int
            outputs per unit and stream (O)
        units : int
            number of units (U)
        streams : int
            number of streams (S)
        """
        self.cores = int(cores)
        self.outputs = int(outputs)
        self.units = int(units)
        self.streams = int(streams)

        self.proj = Conv(channels=self.channels, streams=self.streams).add_input(
            channels=self.cores,
        )
        self.position.init(
            units=self.units,
        )
        self.features.init(
            inputs=self.channels,
            outputs=self.outputs,
            units=self.units,
            streams=self.streams,
        )

        # self.bias = nn.Parameter(
        #     torch.zeros(units),
        # )

    # def _param_groups(self, lr=0.1, decay=0, **kwargs):
    #     yield dict(params=[self.bias], lr=lr * self.units, decay=0, **kwargs)
