import torch
from functools import reduce

from .containers import Module, ModuleList
from .elements import Linear, nonlinearity
from .utils import isotropic_grid_sample_2d, rmat_3d


class Perspective(Module):
    def __init__(
        self,
        stimulus,
        eye_position,
    ):
        """
        Parameters
        ----------
        stimulus : .variables.Stimulus
            stimulus variable
        eye_position : .variables.EyePosition
            eye position variable
        """
        super().__init__()
        self.stimulus = stimulus
        self.eye_position = eye_position

    @property
    def out_channels(self):
        raise NotImplementedError

    def forward(self, stimulus, eye_position=None, height=144, width=256, pad_mode="constant", pad_value=0):
        """
        Parameters
        ----------
        stimulus : Tensor
            shape = [n, c, h, w]
        eye_position : Tensor
            shape = [n, f]
        height : int
            output height, h'
        width : int
            output width, w'
        pad_mode : str
            "constant" | "replicate"
        pad_value : float
            value of padding when pad_mode=="constant"

        Returns
        -------
        Tensor
            shape = [n, c', h', w']
        """
        raise NotImplementedError()

    def inverse(self, stimulus, eye_position=None, height=144, width=256, pad_mode="constant", pad_value=0):
        """
        Parameters
        ----------
        stimulus : Tensor
            shape = [n, c, h, w]
        eye_position : Tensor
            shape = [n, f]
        height : int
            output height, h'
        width : int
            output width, w'
        pad_mode : str
            "constant" | "replicate"
        pad_value : float
            value of padding when pad_mode=="constant"

        Returns
        -------
        Tensor
            shape = [n, c', h', w']
        """
        raise NotImplementedError()


class MonitorRetina(Perspective):
    def __init__(self, stimulus, eye_position, monitor, retina, features, nonlinear=None):
        """
        Parameters
        ----------
        stimulus : .variables.Stimulus
            stimulus variable
        eye_position : .variables.EyePosition
            eye position variable
        monitor : .monitors.Monitor
            3D monitor model
        retina : .retinas.Retina
            3D retina model
        features : Sequence[int]
            MLP features
        nonlinear : str | None
            nonlinearity
        """
        super().__init__(stimulus=stimulus, eye_position=eye_position)
        self.monitor = monitor
        self.retina = retina
        self.features = list(map(int, features))

        in_features = [self.eye_position.n_features, *self.features]
        layers = [Linear(out_features=o).add(in_features=i) for o, i in zip(self.features, in_features)]
        self.layers = ModuleList(layers)

        self.proj = Linear(out_features=3).add(in_features=self.features[-1])
        torch.nn.init.constant_(self.proj.gain, 0)

        self.nonlinear, self.gamma = nonlinearity(nonlinear=nonlinear)

    @property
    def out_channels(self):
        return self.stimulus.n_channels

    def rmat(self, eye_position=None):
        """
        Parameters
        ----------
        eye_position : Tensor | None
            shape = [n, f]

        Returns
        -------
        Tensor
            shape = [n, 3, 3]
        """
        x = self.eye_position(eye_position)
        x = reduce(lambda x, layer: self.nonlinear(layer([x])) * self.gamma, self.layers, x)
        x = self.proj([x])
        return rmat_3d(*x.unbind(1))

    def forward(self, stimulus, eye_position=None, height=144, width=256, pad_mode="constant", pad_value=0):
        """
        Parameters
        ----------
        stimulus : Tensor
            shape = [n, c, h, w]
        eye_position : Tensor
            shape = [n, f]
        height : int
            output height, h'
        width : int
            output width, w'
        pad_mode : str
            "constant" | "replicate"
        pad_value : float
            value of padding when pad_mode=="constant"

        Returns
        -------
        Tensor
            shape = [n, c', h', w']
        """
        rmat = self.rmat(eye_position)
        rays = self.retina.rays(rmat, height, width)
        grid = self.monitor.project(rays)
        out = isotropic_grid_sample_2d(
            stimulus,
            grid=grid,
            pad_mode=pad_mode,
            pad_value=pad_value,
        )
        return self.stimulus(out)

    def inverse(self, stimulus, eye_position=None, height=144, width=256, pad_mode="constant", pad_value=0):
        """
        Parameters
        ----------
        stimulus : Tensor
            shape = [n, c, h, w]
        eye_position : Tensor
            shape = [n, f]
        height : int
            output height, h'
        width : int
            output width, w'
        pad_mode : str
            "constant" | "replicate"
        pad_value : float
            value of padding when pad_mode=="constant"

        Returns
        -------
        Tensor
            shape = [n, c', h', w']
        """
        rays = self.monitor.rays(stimulus.size(0), height, width)
        rmat = self.rmat(eye_position)
        grid = self.retina.project(rays, rmat)
        out = isotropic_grid_sample_2d(
            stimulus,
            grid=grid,
            pad_mode=pad_mode,
            pad_value=pad_value,
        )
        return self.stimulus(out, inverse=True)
