import torch
from .modules import Module
from .elements import Linear, FlatDropout, nonlinearity


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


class LnLstm(Modulation):
    """Linear -> Nonlinear -> Lstm"""

    def __init__(self, features, nonlinear=None, dropout=0):
        """
        Parameters
        ----------
        features : int
            feature size
        nonlinear : str | None
            nonlinearity
        dropout : float
            dropout probability -- [0, 1)
        """
        super().__init__()

        self._features = int(features)
        self.nonlinear, self.gamma = nonlinearity(nonlinear)
        self._dropout = float(dropout)

    def _init(self, modulations):
        """
        Parameters
        ----------
        modulations : int
            number of inputs (I)
        """
        self.modulations = int(modulations)

        self.proj_x = Linear(features=self.features).add_input(features=self.modulations)
        self.proj_i = Linear(features=self.features).add_input(features=self.features * 2)
        self.proj_f = Linear(features=self.features).add_input(features=self.features * 2)
        self.proj_g = Linear(features=self.features).add_input(features=self.features * 2)
        self.proj_o = Linear(features=self.features).add_input(features=self.features * 2)

        self.drop = FlatDropout(p=self._dropout)
        self._past = dict()

    def _restart(self):
        self.dropout(p=self._dropout)

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

        x = self.proj_x([modulation])
        x = self.nonlinear(x) * self.gamma

        xh = torch.cat([x, h], dim=1)

        i = torch.sigmoid(self.proj_i([xh]))
        f = torch.sigmoid(self.proj_f([xh]))
        g = torch.tanh(self.proj_g([xh]))
        o = torch.sigmoid(self.proj_o([xh]))

        c = f * c + i * g
        h = o * torch.tanh(c)
        h = self.drop(h)

        self._past["c"] = c
        self._past["h"] = h

        return h
