import torch

from .containers import Module


class Features(Module):
    def init(self, units, inputs, outputs, streams):
        """
        Parameters
        ----------
        units : int
            number of units, u
        inputs : int
            number of inputs, i
        outputs : int
            number of outputs, o
        streams : int
            number of streams, s
        """
        raise NotImplementedError()

    def weights(self, stream=None):
        """
        Parameters
        ----------
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            shape = [u, i, o] -- stream is None
                or
            shape = [u, i // s, o // s] -- stream is int
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

        self.weights = nn.ParameterList()
        self.gains = nn.ParameterList()

        self._weights = dict()

    def _reset(self):
        self._weights.clear()

    def _param_norm_dims(self):
        for weight in self.weights:
            yield weight, 1

    def _param_groups(self, **kwargs):
        if kwargs.get("weight_decay"):
            kwargs.update(weight_decay=0)
            yield dict(params=list(self.gain), **kwargs)

    # def init(self, units, features):
    #     """
    #     Parameters
    #     ----------
    #     units : int
    #         number of units, u
    #     features : int
    #         number of features, c
    #     """
    #     self.weight = torch.nn.Parameter(torch.ones(units, features))
    #     self.gain = torch.nn.Parameter(torch.ones(units))

    #     bound = features**-0.5
    #     torch.nn.init.uniform_(self.weight, -bound, bound)

    # @property
    # def weights(self):
    #     """
    #     Returns
    #     -------
    #     Tensor
    #         shape = [u, c]
    #     """
    #     if self._weights is None:

    #         var, mean = torch.var_mean(self.weight, dim=1, unbiased=False, keepdim=True)
    #         scale = (var * self.weight.size(dim=1) + self.eps).pow(-0.5)

    #         self._weights = torch.einsum(
    #             "U F , U -> U F",
    #             (self.weight - mean).mul(scale),
    #             self.gain,
    #         )

    #     return self._weights
