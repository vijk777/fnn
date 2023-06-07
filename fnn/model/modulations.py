import torch
from .modules import Module
from .elements import Linear


# -------------- Modulation Prototype --------------


class Modulation(Module):
    """Modulation Model"""

    def _init(self, modulations):
        """
        Parameters
        ----------
        modulations : int
            modulation inputs (I)
        """
        raise NotImplementedError()

    @property
    def features(self):
        """
        Returns
        -------
        int
            modulation features (M)
        """
        raise NotImplementedError()

    def forward(self, modulation):
        """
        Parameters
        ----------
        modulation : Tensor
            [N, I]

        Returns
        -------
        Tensor
            [N, M]
        """
        raise NotImplementedError()


# -------------- Modulation Types --------------


class Lstm(Modulation):
    """Lstm Modulation"""

    def __init__(self, features):
        """
        Parameters
        ----------
        features : int
            lstm features
        """
        super().__init__()

        self._features = int(features)
        self._past = dict()

    def _init(self, modulations):
        """
        Parameters
        ----------
        modulations : int
            number of inputs (I)
        """
        self.modulations = int(modulations)
        self.proj_i = (
            Linear(features=self.features).add_input(features=self.modulations).add_input(features=self.features)
        )
        self.proj_f = (
            Linear(features=self.features).add_input(features=self.modulations).add_input(features=self.features)
        )
        self.proj_g = (
            Linear(features=self.features).add_input(features=self.modulations).add_input(features=self.features)
        )
        self.proj_o = (
            Linear(features=self.features).add_input(features=self.modulations).add_input(features=self.features)
        )

    def _reset(self):
        self._past.clear()

    @property
    def features(self):
        """
        Returns
        -------
        int
            modulation features (M)
        """
        return self._features

    def forward(self, modulation):
        """
        Parameters
        ----------
        modulation : Tensor
            [N, I]

        Returns
        -------
        Tensor
            [N, M]
        """
        if self._past:
            h = self._past["h"]
            c = self._past["c"]
        else:
            h = c = torch.zeros(modulation.size(0), self.features, device=self.device)

        inputs = [modulation, h]

        i = torch.sigmoid(self.proj_i(inputs))
        f = torch.sigmoid(self.proj_f(inputs))
        g = torch.tanh(self.proj_g(inputs))
        o = torch.sigmoid(self.proj_o(inputs))

        c = f * c + i * g
        h = o * torch.tanh(c)

        self._past["c"] = c
        self._past["h"] = h

        return h
