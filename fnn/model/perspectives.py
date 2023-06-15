from torch.nn import init
from functools import reduce
from .modules import Module, ModuleList
from .elements import Linear, nonlinearity
from .utils import isotropic_grid_sample_2d, rmat_3d


# -------------- Perspective Prototype --------------


class Perspective(Module):
    """Perspective Module"""

    def _init(self, stimuli, perspectives):
        """
        Parameters
        ----------
        stimuli : int
            stimulus channels (S)
        perspectives : int
            perspective features (F)
        """
        raise NotImplementedError()

    @property
    def channels(self):
        """
        Returns
        -------
        int
            perspective channels (P)
        """
        raise NotImplementedError()

    def forward(self, stimulus, perspective, pad_mode="constant", pad_value=0):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, S, H, W]
        perspective : Tensor
            [N, F]
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

    def inverse(self, stimulus, perspective, height=144, width=256, pad_mode="constant", pad_value=0):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, S, H, W]
        perspective : Tensor
            [N, F]
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


# -------------- Perspective Types --------------


class MonitorRetina(Perspective):
    """Monitor & Retina Perspective"""

    def __init__(self, monitor, monitor_pixel, retina, retina_pixel, height, width, features, nonlinear=None):
        """
        Parameters
        ----------
        monitor : fnn.model.monitors.Monitor
            3D monitor model
        monitor_pixel : fnn.model.pixels.Pixel
            monitor pixel intensity
        retina : fnn.model.retinas.Retina
            3D retina model
        retina_pixel : fnn.model.pixels.Pixel
            retina pixel intensity
        height : int
            retina height
        width : int
            retina width
        features : Sequence[int]
            mlp features
        nonlinear : str | None
            nonlinearity
        """
        super().__init__()

        self.monitor = monitor
        self.monitor_pixel = monitor_pixel

        self.retina = retina
        self.retina._init(height=height, width=width)
        self.retina_pixel = retina_pixel

        self.features = list(map(int, features))
        self.layers = ModuleList([Linear(features=f) for f in self.features])

        for layer, f in zip(self.layers[1:], self.features):
            layer.add_input(features=f)

        self.proj = Linear(features=3).add_input(
            features=self.features[-1],
        )
        for gain in self.proj.gains:
            init.constant_(gain, 0)

        self.nonlinear, self.gamma = nonlinearity(nonlinear=nonlinear)

    def _init(self, stimuli, perspectives):
        """
        Parameters
        ----------
        stimuli : int
            stimulus channels (S)
        perspectives : int
            perspective features (F)
        """
        self._channels = int(stimuli)
        self.layers[0].add_input(features=perspectives)

    @property
    def channels(self):
        """
        Returns
        -------
        int
            perspective channels (P)
        """
        return self._channels

    def rmat(self, perspective):
        """
        Parameters
        ----------
        perspective : Tensor
            [N, F]

        Returns
        -------
        Tensor
            [N, 3, 3], 3D rotation matrix
        """
        x = reduce(lambda x, layer: self.nonlinear(layer([x])) * self.gamma, self.layers, perspective)
        x = self.proj([x])
        return rmat_3d(*x.unbind(1))

    def forward(self, stimulus, perspective, pad_mode="constant", pad_value=0):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, S, H, W]
        perspective : Tensor
            [N, F]
        pad_mode : str
            "constant" | "replicate"
        pad_value : float
            value of padding when pad_mode=="constant"

        Returns
        -------
        Tensor
            [N, P, H', W']
        """
        size = max(stimulus.size(0), perspective.size(0))

        rmat = self.rmat(perspective).expand(size, -1, -1)
        rays = self.retina.rays(rmat)
        grid = self.monitor.project(rays)

        pixels = self.monitor_pixel(stimulus).expand(size, -1, -1, -1)
        pixels = isotropic_grid_sample_2d(pixels, grid=grid, pad_mode=pad_mode, pad_value=pad_value)
        pixels = self.retina_pixel(pixels)

        return pixels

    def inverse(self, stimulus, perspective, height=144, width=256, pad_mode="constant", pad_value=0):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, S, H, W]
        perspective : Tensor
            [N, F]
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
        size = max(stimulus.size(0), perspective.size(0))

        rmat = self.rmat(perspective).expand(size, -1, -1)
        rays = self.monitor.rays(size, height, width)
        grid = self.retina.project(rays, rmat)

        pixels = self.retina_pixel.inverse(stimulus).expand(size, -1, -1, -1)
        pixels = isotropic_grid_sample_2d(pixels, grid=grid, pad_mode=pad_mode, pad_value=pad_value)
        pixels = self.monitor_pixel.inverse(pixels)

        return pixels
