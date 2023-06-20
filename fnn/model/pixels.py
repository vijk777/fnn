import torch
from .modules import Module


# -------------- Pixel Prototype --------------


class Pixel(Module):
    """Pixel Intensity"""

    def forward(self, pixels):
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

    def inverse(self, pixels):
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


class Raw(Pixel):
    """Raw Pixel Intensity"""

    def forward(self, pixels):
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
        return pixels

    def inverse(self, pixels):
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
        return pixels


class Linear(Pixel):
    """Linear Pixel Intensity"""

    def __init__(self, scale=1, offset=0):
        """
        Parameters
        ----------
        scale : float
            luminance scale
        """
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float))

    def forward(self, pixels):
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
        return pixels.mul(self.scale)

    def inverse(self, pixels):
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
        return pixels.div(self.scale)

    def extra_repr(self):
        return f"scale={self.scale:.3g}"


class Power(Pixel):
    """Linear Pixel Intensity"""

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

    def forward(self, pixels):
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
        return pixels.pow(self.power).mul(self.scale).add(self.offset)

    def inverse(self, pixels):
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
        return pixels.sub(self.offset).div(self.scale).pow(1 / self.power)

    def extra_repr(self):
        return f"power={self.power:.3g}, scale={self.scale:.3g}, offset={self.offset:.3g}"
