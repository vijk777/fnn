import torch
from torch import nn

from .containers import Module
from .elements import Conv
from .utils import to_groups_2d


class Readout(Module):
    def init(self, cores, readouts, units, streams):
        """
        Parameters
        ----------
        cores : int
            core channels per stream (C)
        readouts : int
            readouts per unit and stream (R)
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
            [N, S, U, R] -- stream is None
                or
            [N, U, R] -- stream is int
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

    def _param_groups(self, lr=0.1, decay=0, **kwargs):
        yield dict(params=list(self.biases), lr=lr * self.units, decay=0, **kwargs)

    def init(self, cores, readouts, units, streams):
        """
        Parameters
        ----------
        cores : int
            core channels per stream (C)
        readouts : int
            readouts per unit and stream (O)
        units : int
            number of units (U)
        streams : int
            number of streams (S)
        """
        self.cores = int(cores)
        self.readouts = int(readouts)
        self.units = int(units)
        self.streams = int(streams)

        self.proj = Conv(channels=self.channels, streams=self.streams).add_input(
            channels=self.cores,
            drop=True,
        )
        self.position.init(
            units=self.units,
        )
        self.features.init(
            inputs=self.channels,
            outputs=self.readouts,
            units=self.units,
            streams=self.streams,
        )
        bias = lambda: nn.Parameter(torch.zeros(self.units, self.readouts))
        self.biases = nn.ParameterList([bias() for _ in range(self.streams)])

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
            [N, S, U, R] -- stream is None
                or
            [N, U, R] -- stream is int
        """
        if self.training:
            position = self.position.sample(core.size(0))
        else:
            position = self.position.mean.expand(core.size(0), -1, -1)

        out = nn.functional.grid_sample(
            self.proj([core], stream=stream),
            grid=self.bound(position)[:, :, None, :],
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        features = self.features(stream=stream)

        if stream is None:
            out = to_groups_2d(out, self.streams).squeeze(4)
            out = torch.einsum("S U R C , N S C U -> N S U R", features, out)
        else:
            out = out.squeeze(3)
            out = torch.einsum("U R C , N C U -> N U R", features, out)
