import torch


# -------------- Trainer Prototype --------------


class Trainer:
    """Module Trainer"""

    def epoch(self, optimizer, seed=42):
        """
        Parameters
        ----------
        optimizer : fnn.train.trainers.Optimizer
            module optimizer
        seed : int
            random number generation
        """
        raise NotImplementedError()
