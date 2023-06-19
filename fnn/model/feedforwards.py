import math
from torch.nn import init
from .modules import Module, ModuleList
from .elements import Conv, Residual, StreamDropout, nonlinearity
from .utils import to_groups_2d


# -------------- Feedforward Prototype --------------


class Feedforward(Module):
    """Feature Module"""

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            [input channels per stream (I), ...]
        streams : int
            number of streams, S
        """
        raise NotImplementedError()

    @property
    def channels(self):
        """
        Returns
        -------
        int
            feedforward channels per stream (F)
        """
        raise NotImplementedError()

    def forward(self, inputs, stream=None):
        """
        Parameters
        ----------
        inputs : Sequence[Tensor]
            [[N, S*I, H, W] ...] -- stream is None
                or
            [[N, I, H, W] ...] -- stream is int
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            [N, S*F, H//D, W//D] -- stream is None
                or
            [N, F, H//D, W//D] -- stream is int
        """
        raise NotImplementedError()


# -------------- Feedforward Types --------------


class SpatialResidual(Feedforward):
    """Spatial Residual"""

    def __init__(self, channels, kernel_sizes, strides, nonlinear=None, dropout=0):
        """
        Parameters
        ----------
        channels : Sequence[int]
            layer channels
        kernel_sizes : Sequence[int]
            layer spatial sizes
        strides : Sequence[int]
            layer spatial strides
        nonlinear : str | None
            nonlinearity
        dropout : float
            dropout probability -- [0, 1)
        """
        assert len(channels) == len(kernel_sizes) == len(strides)
        super().__init__()

        self._channels = list(map(int, channels))
        self.kernel_sizes = list(map(int, kernel_sizes))
        self.strides = list(map(int, strides))
        self.nonlinear, self.gamma = nonlinearity(nonlinear)
        self._drop = float(dropout)

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            [input channels per stream (I), ...]
        streams : int
            number of streams, S
        """
        self.inputs = list(map(int, inputs))
        self.streams = int(streams)

        self.spatial = ModuleList([])
        self.residual = ModuleList([])

        _channels = sum(self.inputs)
        for channels, spatial_size, stride in zip(
            self._channels,
            self.kernel_sizes,
            self.strides,
        ):
            spatial = Conv(channels=channels, streams=streams, gain=False, bias=False).add_input(
                channels=_channels,
                kernel_size=spatial_size,
                stride=stride,
            )
            residual = Residual(
                in_channels=_channels,
                out_channels=channels,
                streams=streams,
                stride=stride,
            )

            self.spatial.append(spatial)
            self.residual.append(residual)

            _channels = channels

        self.drop = StreamDropout(p=self._drop, streams=self.streams)

    @property
    def channels(self):
        """
        Returns
        -------
        int
            feedforward channels per stream (F)
        """
        return self._channels[-1]

    def forward(self, inputs, stream=None):
        """
        Parameters
        ----------
        inputs : Sequence[Tensor]
            [[N, S*I, H, W] ...] -- stream is None
                or
            [[N, I, H, W] ...] -- stream is int
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            [N, S*F, H//D, W//D] -- stream is None
                or
            [N, F, H//D, W//D] -- stream is int
        """
        if len(self.inputs) == 1:
            x = inputs[0]
        elif stream is None:
            x = torch.cat([to_groups_2d(_, self.streams) for _ in inputs], 2).flatten(1, 2)
        else:
            x = torch.cat(inputs, 1)

        for spatial, residual in zip(self.spatial, self.residual):
            y = spatial([x], stream=stream)
            z = residual(x, stream=stream)
            x = self.nonlinear(y) * self.gamma + z

        return self.drop(x, stream=stream)


class SpatialTemporalResidual(Feedforward):
    """Spatial Temporal Residual"""

    def __init__(self, channels, spatial_sizes, spatial_strides, temporal_sizes, nonlinear=None, dropout=0):
        """
        Parameters
        ----------
        channels : Sequence[int]
            layer channels
        spatial_sizes : Sequence[int]
            layer spatial sizes
        spatial_strides : Sequence[int]
            layer spatial strides
        temporal_sizes : Sequence[int]
            layer temporal sizes
        nonlinear : str | None
            nonlinearity
        dropout : float
            dropout probability -- [0, 1)
        """
        assert len(channels) == len(spatial_sizes) == len(temporal_sizes) == len(spatial_strides)
        super().__init__()

        self._channels = list(map(int, channels))
        self.spatial_sizes = list(map(int, spatial_sizes))
        self.spatial_strides = list(map(int, spatial_strides))
        self.temporal_sizes = list(map(int, temporal_sizes))
        self.nonlinear, self.gamma = nonlinearity(nonlinear)
        self._drop = float(dropout)

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            [input channels per stream (I), ...]
        streams : int
            number of streams, S
        """
        self.inputs = list(map(int, inputs))
        self.streams = int(streams)

        self.spatial = ModuleList([])
        self.temporal = ModuleList([])
        self.residual = ModuleList([])

        _channels = sum(self.inputs)
        for channels, spatial_size, temporal_size, stride in zip(
            self._channels,
            self.spatial_sizes,
            self.temporal_sizes,
            self.spatial_strides,
        ):
            spatial = Conv(channels=channels, streams=streams, gain=False, bias=False).add_input(
                channels=_channels,
                kernel_size=spatial_size,
                stride=stride,
            )
            temporal = Conv(channels=channels, streams=streams, gain=True, bias=True).add_input(
                channels=channels,
                dynamic_size=temporal_size,
            )
            residual = Residual(in_channels=_channels, out_channels=channels, streams=streams, stride=stride)

            self.spatial.append(spatial)
            self.temporal.append(temporal)
            self.residual.append(residual)

            _channels = channels

        self.drop = StreamDropout(p=self._drop, streams=self.streams)

    @property
    def channels(self):
        """
        Returns
        -------
        int
            feedforward channels per stream (F)
        """
        return self._channels[-1]

    def forward(self, inputs, stream=None):
        """
        Parameters
        ----------
        inputs : Sequence[Tensor]
            [[N, S*I, H, W] ...] -- stream is None
                or
            [[N, I, H, W] ...] -- stream is int
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            [N, S*F, H//D, W//D] -- stream is None
                or
            [N, F, H//D, W//D] -- stream is int
        """
        if len(self.inputs) == 1:
            x = inputs[0]
        elif stream is None:
            x = torch.cat([to_groups_2d(_, self.streams) for _ in inputs], 2).flatten(1, 2)
        else:
            x = torch.cat(inputs, 1)

        for spatial, temporal, residual in zip(self.spatial, self.temporal, self.residual):
            conv = spatial([x], stream=stream)
            conv = temporal([conv], stream=stream)
            x = self.nonlinear(conv) * self.gamma + residual(x, stream=stream)

        return self.drop(x, stream=stream)
