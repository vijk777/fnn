import math
from torch.nn import init
from .modules import Module, ModuleList
from .elements import Conv, nonlinearity


# -------------- Feedforward Prototype --------------


class Feedforward(Module):
    """Feature Module"""

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        channels : Sequence[[int, bool]]
            [[input channels per stream (I), whether to drop input] ...]
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


# -------------- Feature Prototype --------------


class Res3d(Feedforward):
    """3D Residual"""

    def __init__(self, channels, spatial_sizes, spatial_strides, temporal_sizes, nonlinear=None):
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
        """
        assert len(channels) == len(spatial_sizes) == len(temporal_sizes) == len(spatial_strides)
        super().__init__()

        self._channels = list(map(int, channels))
        self.spatial_sizes = list(map(int, spatial_sizes))
        self.spatial_strides = list(map(int, spatial_strides))
        self.temporal_sizes = list(map(int, temporal_sizes))
        self.nonlinear, self.gamma = nonlinearity(nonlinear)

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        channels : Sequence[[int, bool]]
            [[input channels per stream (I), whether to drop input] ...]
        streams : int
            number of streams, S
        """
        self.inputs = list([int(inp), bool(drop)] for inp, drop in inputs)
        self.streams = int(streams)

        conv = Conv(channels=self._channels[0], streams=streams)
        res = Conv(channels=self._channels[0], streams=streams)

        for _channels, drop in self.inputs:
            conv.add_input(
                channels=_channels,
                drop=drop,
                kernel_size=self.spatial_sizes[0],
                dynamic_size=self.temporal_sizes[0],
                stride=self.spatial_strides[0],
            )
            res.add_input(
                channels=_channels,
                kernel_size=self.spatial_strides[0],
                stride=self.spatial_strides[0],
            )

        self.conv = ModuleList([conv])
        self.residual = ModuleList([res])

        _channels = self._channels[0]
        for channels, spatial_size, temporal_size, stride in zip(
            self._channels[1:],
            self.spatial_sizes[1:],
            self.temporal_sizes[1:],
            self.spatial_strides[1:],
        ):
            conv = Conv(channels=channels, streams=streams).add_input(
                channels=_channels,
                kernel_size=spatial_size,
                dynamic_size=temporal_size,
                stride=stride,
            )
            res = Conv(channels=channels, streams=streams).add_input(
                channels=_channels,
                kernel_size=stride,
                stride=stride,
            )
            self.conv.append(conv)
            self.residual.append(res)
            _channels = channels

        for res in self.residual:
            for gain in res.gains:
                init.constant_(gain, 0)

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
        for conv, res in zip(self.conv, self.residual):

            c = conv(inputs, stream=stream)
            r = res(inputs, stream=stream)
            inputs = [self.nonlinear(c) * self.gamma + r]

        return inputs[0]


class SpatialTemporalResidual(Feedforward):
    """Spatial Temporal Residual"""

    def __init__(self, channels, spatial_sizes, spatial_strides, temporal_sizes, nonlinear=None):
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
        """
        assert len(channels) == len(spatial_sizes) == len(temporal_sizes) == len(spatial_strides)
        super().__init__()

        self._channels = list(map(int, channels))
        self.spatial_sizes = list(map(int, spatial_sizes))
        self.spatial_strides = list(map(int, spatial_strides))
        self.temporal_sizes = list(map(int, temporal_sizes))
        self.nonlinear, self.gamma = nonlinearity(nonlinear)

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        channels : Sequence[[int, bool]]
            [[input channels per stream (I), whether to drop input] ...]
        streams : int
            number of streams, S
        """
        self.inputs = list([int(inp), bool(drop)] for inp, drop in inputs)
        self.streams = int(streams)

        spatial = Conv(channels=self._channels[0], streams=streams, gain=False, bias=False)
        residual = Conv(channels=self._channels[0], streams=streams, gain=True, bias=True)

        for _channels, drop in self.inputs:
            spatial.add_input(
                channels=_channels,
                drop=drop,
                kernel_size=self.spatial_sizes[0],
                stride=self.spatial_strides[0],
            )
            residual.add_input(
                channels=_channels,
                kernel_size=self.spatial_strides[0],
                stride=self.spatial_strides[0],
            )

        temporal = Conv(channels=self._channels[0], streams=streams, gain=True, bias=True).add_input(
            channels=self._channels[0],
            dynamic_size=self.temporal_sizes[0],
        )

        self.spatial = ModuleList([spatial])
        self.temporal = ModuleList([temporal])
        self.residual = ModuleList([residual])

        _channels = self._channels[0]
        for channels, spatial_size, temporal_size, stride in zip(
            self._channels[1:],
            self.spatial_sizes[1:],
            self.temporal_sizes[1:],
            self.spatial_strides[1:],
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
            residual = Conv(channels=channels, streams=streams, gain=True, bias=True).add_input(
                channels=_channels,
                kernel_size=stride,
                stride=stride,
            )
            self.spatial.append(spatial)
            self.temporal.append(temporal)
            self.residual.append(residual)
            _channels = channels

        for residual in self.residual:
            for gain in residual.gains:
                init.constant_(gain, 0)

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
        for spatial, temporal, residual in zip(self.spatial, self.temporal, self.residual):

            conv = spatial(inputs, stream=stream)
            conv = temporal([conv], stream=stream)
            res = residual(inputs, stream=stream)
            inputs = [self.nonlinear(conv) * self.gamma + res]

        return inputs[0]
