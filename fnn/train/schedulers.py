import numpy as np


# -------------- Scheduler Prototype --------------


class Scheduler:
    """Hyperparameter Scheduler"""

    def _init(self, size, epoch=0, cycle=0):
        """
        Parameters
        ----------
        size : int
            epochs in a cycle [1, inf)
        epoch : int
            epoch number [0, inf)
        cycle : int
            cycle number [0, inf)
        """
        assert size > 0
        assert epoch >= 0
        assert cycle >= 0

        self.size = int(size)
        self.epoch = int(epoch) - 1
        self.cycle = int(cycle)

    def step(self):
        """Perform an epoch step

        Returns
        -------
        bool
            whether the cycle is unfinished
        """
        if self.epoch >= self.size:
            raise RuntimeError("cycle has already completed")

        self.epoch += 1
        return self.epoch < self.size

    def __call__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
            hyperparameters to transform

        Returns
        -------
        dict
            transformed hyperparameters
        """
        return kwargs


# -------------- Scheduler Types --------------


class CosineLr(Scheduler):
    """Cosine Learning Rate"""

    def __init__(self, burnin=0, burnin_cycles=0):
        """
        Parameters
        ----------
        burnin : int
            number of burnin epochs
        burnin_cycles : int
            number of burnin cycles
        """
        assert burnin >= 0

        if burnin:
            assert burnin_cycles > 0
        else:
            assert burnin_cycles == 0

        self.burnin = int(burnin)
        self.burnin_cycles = int(burnin_cycles)

    def __call__(self, lr, **kwargs):
        """
        Parameters
        ----------
        lr : float
            learning rate
        **kwargs
            other hyperparameters

        Returns
        -------
        dict
            hyperparameters with transformed learning rate
        """
        burnin_cycle = self.cycle < self.burnin_cycles
        burnin_epoch = self.epoch < self.burnin

        if burnin_cycle and burnin_epoch:
            lr = lr * (self.epoch + 0.5) / self.burnin

        else:
            if burnin_cycle:
                t = self.epoch - self.burnin
                tmax = self.size - self.burnin
            else:
                t = self.epoch
                tmax = self.size

            lr = lr * (np.cos(t / tmax * np.pi) + 1) / 2

        return dict(lr=lr, **kwargs)
