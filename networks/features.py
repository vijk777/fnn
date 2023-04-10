import torch
from torch import nn

from .containers import Module


class Features(Module):
    def init(self, units, inputs, outputs, streams):
        """
        Parameters
        ----------
        units : int
            number of units, u
        inputs : int
            number of inputs per stream, i
        outputs : int
            number of outputs per stream, o
        streams : int
            number of streams, s
        """
        raise NotImplementedError()

    def forward(self, stream=None):
        """
        Parameters
        ----------
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            shape = [s, u, o, i] -- stream is None
                or
            shape = [u, o, i] -- stream is int
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
        self._features = dict()

    def _reset(self):
        self._features.clear()

    def _param_norm_dims(self):
        for weight in self.weights:
            yield weight, 2

    def _param_groups(self, **kwargs):
        if kwargs.get("weight_decay"):
            kwargs.update(weight_decay=0)
            yield dict(params=list(self.gains), **kwargs)

    def init(self, units, inputs, outputs, streams):
        """
        Parameters
        ----------
        units : int
            number of units, u
        inputs : int
            number of inputs per stream, i
        outputs : int
            number of outputs per stream, o
        streams : int
            number of streams, s
        """
        self.inputs = int(inputs)
        self.outputs = int(outputs)
        self.streams = int(streams)

        weight = lambda: nn.Parameter(torch.ones(units, outputs, inputs))
        gain = lambda: nn.Parameter(torch.ones(units, outputs))

        self.weights = nn.ParameterList([weight() for _ in range(streams)])
        self.gains = nn.ParameterList([gain() for _ in range(streams)])

        bound = inputs**-0.5
        for weight in self.weights:
            nn.init.uniform_(weight, -bound, bound)

    def features(self, stream):
        """
        Parameters
        ----------
        stream : int
            stream index

        Returns
        -------
        Tensor
            shape = [u, o, i]
        """
        features = self._features.get(stream)

        if features is None:
            weight = self.weights[stream]
            gain = self.gains[stream]

            var, mean = torch.var_mean(weight, dim=2, unbiased=False, keepdim=True)
            scale = (var * self.inputs + self.eps).pow(-0.5)

            features = torch.einsum("U O I , U O -> U O I", (weight - mean) * scale, gain)
            self._features[stream] = features

        return features

    def forward(self, stream=None):
        """
        Parameters
        ----------
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            shape = [s, u, o, i] -- stream is None
                or
            shape = [u, o, i] -- stream is int
        """
        if stream is None:
            features = [self.features(stream=s) for s in range(self.streams)]
            features = torch.stack(features, dim=0)

        else:
            features = self.features(stream=stream)

        return features
