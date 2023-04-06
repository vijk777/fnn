import torch

from .containers import Module


class Position(Module):
    def init(self, units):
        """
        Parameters
        ----------
        units : int
            number of units, u
        """
        raise NotImplementedError()

    def forward(self, batch_size=1):
        """
        Parameters
        ----------
        batch_size : int
            batch size, n

        Returns
        -------
        Tensor
            shape = [n, u, 2]
        """
        raise NotImplementedError()


class Gaussian(Position):
    def __init__(self, init_std=0.4):
        super().__init__()
        self.init_std = float(init_std)
        self._position = None

    def _reset(self):
        self._position = None

    def init(self, units):
        """
        Parameters
        ----------
        units : int
            number of units, u
        """
        self.mean = torch.nn.Parameter(torch.zeros(units, 2))
        self.std = torch.nn.Parameter(torch.eye(2).repeat(units, 1, 1))
        self._restart()

    def _restart(self):
        with torch.no_grad():
            self.std.copy_(torch.eye(2).mul(self.init_std))

    def _param_groups(self, **kwargs):
        if kwargs.get("weight_decay"):
            kwargs.update(weight_decay=0)
            yield dict(params=[self.mean, self.std], **kwargs)

    def forward(self, batch_size=1):
        """
        Parameters
        ----------
        batch_size : int
            batch size, n

        Returns
        -------
        Tensor
            shape = [n, u, 2]
        """
        if self._position is None:
            x = self.mean.repeat(batch_size, 1, 1)

            if self.training:
                x = x + torch.einsum("U C D , N U D -> N U C", self.std, torch.randn_like(x))

            self._position = x

        else:
            assert batch_size == self._position.size(0)

        return self._position
