import numpy as np


# -------------- Scheduler Prototype --------------


class Scheduler:
    """Hyperparameter Scheduler"""

    def _init(self, epoch=0, cycle=0):
        """
        Parameters
        ----------
        epoch : int
            epoch number
        cycle : int
            cycle number
        """
        assert epoch >= 0
        assert cycle >= 0

        self.epoch = int(epoch) - 1
        self.cycle = int(cycle)

    def step(self):
        """Perform an epoch step

        Returns
        -------
        bool
            whether the cycle is unfinished
        """
        if self.finished:
            raise RuntimeError("training cycle has already finished")

        self.epoch += 1
        return not self.finished

    @property
    def finished(self):
        """
        Returns
        -------
        bool
            whether the training cycle is finished
        """
        raise NotImplementedError()

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

    def __init__(self, cycle_size=100, burnin_epochs=0, burnin_cycles=0):
        """
        Parameters
        ----------
        cycle_size : int
            number of epochs in a cycle
        burnin : int
            number of burnin epochs
        burnin_cycles : int
            number of burnin cycles
        """
        assert cycle_size > 0
        assert burnin_epochs >= 0

        if burnin_epochs:
            assert burnin_cycles > 0
        else:
            assert burnin_cycles == 0

        self.cycle_size = int(cycle_size)
        self.burnin_epochs = int(burnin_epochs)
        self.burnin_cycles = int(burnin_cycles)

    @property
    def finished(self):
        return self.epoch >= self.cycle_size

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
        burnin_epoch = self.epoch < self.burnin_epochs

        if burnin_cycle and burnin_epoch:
            lr = lr * (self.epoch + 0.5) / self.burnin_epochs

        else:
            if burnin_cycle:
                t = self.epoch - self.burnin_epochs
                tmax = self.cycle_size - self.burnin_epochs
            else:
                t = self.epoch
                tmax = self.cycle_size

            lr = lr * (np.cos(t / tmax * np.pi) + 1) / 2

        return dict(lr=lr, **kwargs)
