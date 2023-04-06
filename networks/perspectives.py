import torch
from functools import reduce
from typing import Optional, Sequence

from .containers import Module, ModuleList
from .variables import Stimulus, EyePosition
from .monitors import Monitor
from .retinas import Retina
from .elements import Linear, nonlinearity
from .utils import isotropic_grid_sample_2d, rmat_3d


class Perspective(Module):
    def __init__(self, stimulus: Stimulus, eye_position: EyePosition):
        super().__init__()
        self.stimulus = stimulus
        self.eye_position = eye_position

    @property
    def out_channels(self):
        raise NotImplementedError

    def forward(
        self,
        stimulus: torch.Tensor,
        eye_position: Optional[torch.Tensor] = None,
        height: int = 144,
        width: int = 256,
        pad_mode: str = "constant",
        pad_value: float = 0,
    ):
        """
        Args:
            stimulus        (torch.Tensor)  : shape = [n, c, h, w]
            eye_position    (torch.Tensor)  : shape = [n, f]
            height          (int)           : h'
            width           (int)           : w'
            pad_mode        (str)           : 'constant' or 'replicate'
            pad_value       (float)         : value of padding when pad_mode=='constant'
        Returns:
            (torch.Tensor)                  : shape = [n, c', h', w']
        """
        raise NotImplementedError()

    def inverse(
        self,
        stimulus: torch.Tensor,
        eye_position: Optional[torch.Tensor] = None,
        height: int = 144,
        width: int = 256,
        pad_mode: str = "constant",
        pad_value: float = 0,
    ):
        """
        Args:
            stimulus        (torch.Tensor)  : shape = [n, c, h, w]
            eye_position    (torch.Tensor)  : shape = [n, f]
            height          (int)           : h'
            width           (int)           : w'
            pad_mode        (str)           : 'constant' or 'replicate'
            pad_value       (float)         : value of padding when pad_mode=='constant'
        Returns:
            (torch.Tensor)                  : shape = [n, c', h', w']
        """
        raise NotImplementedError()


class MonitorRetina(Perspective):
    def __init__(
        self,
        stimulus: Stimulus,
        eye_position: EyePosition,
        monitor: Monitor,
        retina: Retina,
        features: Sequence[int],
        nonlinear: Optional[str] = None,
    ):
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

    def rmat(self, eye_position: Optional[torch.Tensor] = None):
        """
        Args:
            eye_position    (torch.Tensor)  : shape = [n, f]
        Returns:
            (torch.Tensor)                  : shape = [n, 3, 3]
        """
        x = self.eye_position(eye_position)
        x = reduce(lambda x, layer: self.nonlinear(layer([x])) * self.gamma, self.layers, x)
        x = self.proj([x])
        return rmat_3d(*x.unbind(1))

    def forward(
        self,
        stimulus: torch.Tensor,
        eye_position: Optional[torch.Tensor] = None,
        height: int = 144,
        width: int = 256,
        pad_mode: str = "constant",
        pad_value: float = 0,
    ):
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

    def inverse(
        self,
        stimulus: torch.Tensor,
        eye_position: Optional[torch.Tensor] = None,
        height: int = 144,
        width: int = 256,
        pad_mode: str = "constant",
        pad_value: float = 0,
    ):
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
