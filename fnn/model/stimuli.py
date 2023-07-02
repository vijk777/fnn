import torch
from .modules import Module


# -------------- Stimulus Base --------------


class Stimulus(Module):
    """Stimulus Module"""

    def _init(self, channels, timepoints):
        """
        Parameters
        ----------
        channels : int
            number of channels (C)
        timepoints : int
            number of timepoints (T)
        """
        raise NotImplementedError()


# -------------- Stimulus Types --------------


class VisualStimulus(Stimulus):
    """Visual Stimulus"""

    def __init__(self, height, width, bound):
        """
        Parameters
        ----------
        height : int
            stimulus height (H)
        width : int
            stimulus width (W)
        bound : fnn.model.bounds.Bound
            stimulus bound
        """
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.bound = bound
