import torch
from .modules import Module


# -------------- Luminance Prototype --------------


class Luminance(Module):
    """Luminance Module"""

    def forward(self, stimulus):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, C, H, W]

        Returns
        -------
        Tensor
            [N, C, H, W]
        """
        raise NotImplementedError()


# -------------- Luminance Types --------------


class Linear(Luminance):
    def __init__(self, scale=1, offset=0):
        """
        Parameters
        ----------
        scale : float
            luminance scale
        offset : float
            luminance offset
        """
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float))
        self.register_buffer("offset", torch.tensor(offset, dtype=torch.float))

    def forward(self, stimulus):
        return stimulus.mul(self.scale).add(self.offset)

    def inverse(self, stimulus):
        return stimulus.sub(self.offset).div(self.scale)


class Power(Luminance):
    def __init__(self, power=1, scale=1, offset=0):
        """
        Parameters
        ----------
        power : float
            luminance power
        scale : float
            luminance scale
        offset : float
            luminance offset
        """
        super().__init__()
        self.register_buffer("power", torch.tensor(power, dtype=torch.float))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float))
        self.register_buffer("offset", torch.tensor(offset, dtype=torch.float))

    def forward(self, stimulus):
        return stimulus.pow(self.power).mul(self.scale).add(self.offset)

    def inverse(self, stimulus):
        return stimulus.sub(self.offset).div(self.scale).pow(1 / self.power)
