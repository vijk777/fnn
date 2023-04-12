import torch
from torch import nn
from functools import reduce

from .modules import Module, ModuleList
from .elements import Linear, nonlinearity
from .utils import isotropic_grid_sample_2d, rmat_3d


class Perspective(Module):
    @property
    def channels(self):
        """
        Returns
        -------
        int
            perspective channels (P)
        """
        raise NotImplementedError()

    def init(self, channels, eye_positions):
        """
        Parameters
        ----------
        channels : int
            stimulus channels (S)
        eye_positions : int
            eye position features (E)
        """
        raise NotImplementedError()

    def forward(self, stimulus, eye_position, height=144, width=256, pad_mode="constant", pad_value=0):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, S, H, W]
        eye_position : Tensor
            [N, E]
        height : int
            output height (H')
        width : int
            output width (W')
        pad_mode : str
            "constant" | "replicate"
        pad_value : float
            value of padding when pad_mode=="constant"

        Returns
        -------
        Tensor
            [N, P, H', W']
        """
        raise NotImplementedError()

    def inverse(self, stimulus, eye_position, height=144, width=256, pad_mode="constant", pad_value=0):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, S, H, W]
        eye_position : Tensor
            [N, E]
        height : int
            output height (H')
        width : int
            output width (W')
        pad_mode : str
            "constant" | "replicate"
        pad_value : float
            value of padding when pad_mode=="constant"

        Returns
        -------
        Tensor
            [N, P, H', W']
        """
        raise NotImplementedError()


class MonitorRetina(Perspective):
    def __init__(self, monitor, retina, features, nonlinear=None):
        """
        Parameters
        ----------
        monitor : .monitors.Monitor
            3D monitor model
        retina : .retinas.Retina
            3D retina model
        features : Sequence[int]
            MLP features
        nonlinear : str | None
            nonlinearity
        """
        super().__init__()
        self.monitor = monitor
        self.retina = retina
        self.features = list(map(int, features))
        self.layers = ModuleList([Linear(out_features=f) for f in self.features])

        for layer, f in zip(self.layers[1:], self.features):
            layer.add(in_features=f)

        self.proj = Linear(out_features=3).add(in_features=self.features[-1])
        nn.init.constant_(self.proj.gains, 0)

        self.nonlinear, self.gamma = nonlinearity(nonlinear=nonlinear)

    @property
    def channels(self):
        """
        Returns
        -------
        int
            perspective channels (P)
        """
        return self._channels

    def init(self, channels, eye_positions):
        """
        Parameters
        ----------
        channels : int
            stimulus channels (S)
        eye_positions : int
            eye position features (E)
        """
        self._channels = int(channels)
        self.layers[0].add(in_features=eye_positions)

    def rmat(self, eye_position):
        """
        Parameters
        ----------
        eye_position : Tensor
            [N, E]

        Returns
        -------
        Tensor
            [N, 3, 3], 3D rotation matrix
        """
        x = reduce(lambda x, layer: self.nonlinear(layer([x])) * self.gamma, self.layers, eye_position)
        x = self.proj([x])
        return rmat_3d(*x.unbind(1))

    def forward(self, stimulus, eye_position, height=144, width=256, pad_mode="constant", pad_value=0):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, S, H, W]
        eye_position : Tensor
            [N, E]
        height : int
            output height (H')
        width : int
            output width (W')
        pad_mode : str
            "constant" | "replicate"
        pad_value : float
            value of padding when pad_mode=="constant"

        Returns
        -------
        Tensor
            [N, P, H', W']
        """
        rmat = self.rmat(eye_position)
        rays = self.retina.rays(rmat, height, width)
        grid = self.monitor.project(rays)
        return isotropic_grid_sample_2d(
            stimulus,
            grid=grid,
            pad_mode=pad_mode,
            pad_value=pad_value,
        )

    def inverse(self, stimulus, eye_position=None, height=144, width=256, pad_mode="constant", pad_value=0):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, S, H, W]
        eye_position : Tensor
            [N, E]
        height : int
            output height (H')
        width : int
            output width (W')
        pad_mode : str
            "constant" | "replicate"
        pad_value : float
            value of padding when pad_mode=="constant"

        Returns
        -------
        Tensor
            [N, P, H', W']
        """
        rays = self.monitor.rays(stimulus.size(0), height, width)
        rmat = self.rmat(eye_position)
        grid = self.retina.project(rays, rmat)
        return isotropic_grid_sample_2d(
            stimulus,
            grid=grid,
            pad_mode=pad_mode,
            pad_value=pad_value,
        )
