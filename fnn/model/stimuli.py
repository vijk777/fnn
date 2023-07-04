import torch
from .modules import Module
from .parameters import Parameter
from .utils import Gaussian3d


# -------------- Visual Stimulus Base --------------


class VisualStimulus(Stimulus):
    """Visual Stimulus"""

    def forward(self):
        """
        Yields
        ------
        3D Tensor
            [C, H, W] -- stimulus frame
        """
        raise NotImplementedError()

    def penalty(self):
        """
        Returns
        -------
        Tensor
            scalar -- penalty value
        """
        raise NotImplementedError()


# -------------- Visual Stimulus Types --------------


class VisualNlm(VisualStimulus):
    """Visual Stimulus with Non-local means Regularization"""

    def __init__(self, bound, spatial_std=1, temporal_std=1, cutoff=4):
        """
        Parameters
        ----------
        bound : fnn.model.bounds.Bound
            stimulus bound
        spatial_sigma : float
            nlm spatial standard deviation
        temporal_sigma : float
            nlm temporal standard deviation
        cutoff : float
            nlm standard deviation cutoff
        """
        super().__init__()
        self.bound = bound
        self.gaussian = Gaussian3d(spatial_std=spatial_std, temporal_std=temporal_std, cutoff=cutoff)

    def _init(self, channels, timepoints, height, width):
        """
        Parameters
        ----------
        channels : int
            number of channels (C)
        timepoints : int
            number of timepoints (T)
        height : int
            height in pixels (H)
        width : int
            width in pixels (W)
        """
        self.channels = int(channels)
        self.timepoints = int(timepoints)
        self.height = int(height)
        self.width = int(width)
        self.frames = Parameter(torch.zeros([self.channels, self.timepoints, self.height, self.width]))

    def forward(self):
        """
        Yields
        ------
        3D Tensor
            [C, H, W] -- stimulus frame
        """
        yield from self.bound(self.frames).unbind(dim=1)

    def penalty(self):
        """https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/Buades-NonLocal.pdf

        Returns
        -------
        Tensor
            scalar -- penalty value
        """
        delt = self.frames - self.gaussian(self.frames)
        return delt.pow(2).sum()
