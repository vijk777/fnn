import numpy as np
import torch.distributed as dist


# -------------- Scheduler Base --------------


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
    def seed(self):
        """
        Returns
        -------
        int
            training seed
        """
        raise NotImplementedError()

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

    def __init__(self, cycle_size=100, warmup_epochs=0, warmup_cycles=0):
        """
        Parameters
        ----------
        cycle_size : int
            number of epochs in a cycle
        warmup_epochs : int
            number of warmup epochs
        warmup_cycles : int
            number of warmup cycles
        """
        assert cycle_size > 0
        assert warmup_epochs >= 0

        if warmup_epochs:
            assert warmup_cycles > 0
        else:
            assert warmup_cycles == 0

        self.cycle_size = int(cycle_size)
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_cycles = int(warmup_cycles)

    @property
    def seed(self):
        """
        Returns
        -------
        int
            training seed
        """
        if dist.is_initialized():
            rank = dist.get_rank()
            size = dist.get_world_size()
        else:
            rank = 0
            size = 1

        return rank + size * (self.epoch + self.cycle_size * self.cycle)

    @property
    def finished(self):
        """
        Returns
        -------
        bool
            whether the training cycle is finished
        """
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
        warmup_cycle = self.cycle < self.warmup_cycles
        warmup_epoch = self.epoch < self.warmup_epochs

        if warmup_cycle and warmup_epoch:
            lr = lr * (self.epoch + 0.5) / self.warmup_epochs

        else:
            if warmup_cycle:
                t = self.epoch - self.warmup_epochs
                tmax = self.cycle_size - self.warmup_epochs
            else:
                t = self.epoch
                tmax = self.cycle_size

            lr = lr * (np.cos(t / tmax * np.pi) + 1) / 2

        return dict(lr=lr, **kwargs)
