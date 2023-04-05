import torch
from torch import nn

from .containers import Module
from .utils import isotropic_grid_2d, rmat_3d


class Monitor(Module):
    def project(self, rays: torch.Tensor):
        """
        Args:
            rays (torch.Tensor) : shape = [n, h, w, 3]
        Returns:
            (torch.Tensor)      : shape = [n, h, w, 2]
        """
        raise NotImplementedError()

    def rays(self, batch_size: int = 1, height: int = 144, width: int = 256):
        """
        Args:
            batch_size  (int)
            height      (int)
            width       (int)
        Returns:
            (torch.Tensor)      : shape = [batch_size, height, width, 3]
        """
        raise NotImplementedError()


class Plane(Monitor):
    def __init__(
        self,
        init_center: float = 0.5,
        init_center_std: float = 0.05,
        init_angle_std: float = 0.05,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.init_center = float(init_center)
        self.init_center_std = float(init_center_std)
        self.init_angle_std = float(init_angle_std)
        self.eps = float(eps)

        self.center = nn.Parameter(torch.tensor([0, 0, self.init_center]))
        self.angle = nn.Parameter(torch.zeros(3))

        self.center_std = nn.Parameter(torch.zeros(3))
        self.angle_std = nn.Parameter(torch.zeros(3))
        self._restart()

        self._position = dict()

    def _reset(self):
        self._position.clear()

    def _restart(self):
        with torch.no_grad():
            self.center_std.fill_(self.init_center_std)
            self.angle_std.fill_(self.init_angle_std)

    def _param_groups(self, **kwargs):
        if kwargs.get("weight_decay"):
            kwargs.update(weight_decay=0)
            yield dict(params=list(self.parameters()), **kwargs)

    def position(self, batch_size: int = 1):
        """
        Args:
            batch_size  (int)
        Returns:
            center      (torch.Tensor)  : shape = [batch_size, 1, 1, 3]
            X           (torch.Tensor)  : shape = [batch_size, 1, 1, 3]
            Y           (torch.Tensor)  : shape = [batch_size, 1, 1, 3]
            Z           (torch.Tensor)  : shape = [batch_size, 1, 1, 3]
        """
        if self._position:
            assert batch_size == self._position["center"].size(0)

        else:
            center = self.center.repeat(batch_size, 1)
            angle = self.angle.repeat(batch_size, 1)

            if self.training:
                center = center + torch.randn_like(center) * self.center_std
                angle = angle + torch.randn_like(angle) * self.angle_std

            X, Y, Z = rmat_3d(*angle.unbind(1)).unbind(2)

            self._position["center"] = center
            self._position["X"] = X
            self._position["Y"] = Y
            self._position["Z"] = Z

        return self._position["center"], self._position["X"], self._position["Y"], self._position["Z"]

    def project(self, rays: torch.Tensor):
        """
        Args:
            rays (torch.Tensor) : shape = [n, h, w, 3]
        Returns:
            (torch.Tensor)      : shape = [n, h, w, 2]
        """
        center, X, Y, Z = self.position(batch_size=rays.size(0))

        a = torch.einsum("N D , N D -> N", Z, center)[:, None, None]
        b = torch.einsum("N D , N H W D -> N H W", Z, rays).clip(self.eps)

        c = torch.einsum("N H W , N H W D -> N H W D", a / b, rays)
        d = c - center[:, None, None, :]

        proj = [
            torch.einsum("N H W D , N D -> N H W", d, X),
            torch.einsum("N H W D , N D -> N H W", d, Y),
        ]
        return torch.stack(proj, dim=3)

    def rays(self, batch_size: int = 1, height: int = 144, width: int = 256):
        """
        Args:
            batch_size  (int)
            height      (int)
            width       (int)
        Returns:
            (torch.Tensor)      : shape = [batch_size, height, width, 3]
        """
        x, y = isotropic_grid_2d(
            height=height,
            width=width,
            device=self.device,
        ).unbind(2)

        center, X, Y, _ = self.position(batch_size=batch_size)

        X = torch.einsum("H W , N D -> N H W D", x, X)
        Y = torch.einsum("H W , N D -> N H W D", y, Y)

        return center[:, None, None, :] + X + Y

    def extra_repr(self):
        params = self.angle.tolist() + self.center.tolist()
        return "center=[{:.3f}, {:.3f}, {:.3f}], angle=[{:.3f}, {:.3f}, {:.3f}]".format(*params)
