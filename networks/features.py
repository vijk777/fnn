import torch

from .containers import Module


class Features(Module):
    def init(self, units, features):
        """
        Parameters
        ----------
        units : int
            number of units, u
        features : int
            number of features, f
        """
        raise NotImplementedError()

    @property
    def features(self):
        """
        Returns
        -------
        Tensor
            shape = [u, f]
        """
        raise NotImplementedError()


class Standard(Features):
    def __init__(self, eps=1e-5):
        """
        Parameters
        ----------
        eps : float
            small value added to denominator for numerical stability
        """
        super().__init__()
        self.eps = float(eps)
        self._features = None

    def _reset(self):
        self._features = None

    def init(self, units, features):
        """
        Parameters
        ----------
        units : int
            number of units, u
        features : int
            number of features, f
        """
        self.weight = torch.nn.Parameter(torch.ones(units, features))
        self.gain = torch.nn.Parameter(torch.ones(units))

        bound = features**-0.5
        torch.nn.init.uniform_(self.weight, -bound, bound)

    def _param_norm_dims(self):
        yield self.weight, 1
        yield self.gain, 1

    def _param_groups(self, **kwargs):
        if kwargs.get("weight_decay"):
            kwargs.update(weight_decay=0)
            yield dict(params=[self.gain], **kwargs)

    @property
    def features(self):
        """
        Returns
        -------
        Tensor
            shape = [u, f]
        """
        if self._features is None:

            var, mean = torch.var_mean(self.weight, dim=1, unbiased=False, keepdim=True)
            scale = (var * self.weight.size(dim=1) + self.eps).pow(-0.5)

            self._features = torch.einsum(
                "U F , U -> U F",
                (self.weight - mean).mul(scale),
                self.gain,
            )

        return self._features
