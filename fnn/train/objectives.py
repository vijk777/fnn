import torch
import numpy as np
from fnn.model import architectures


# -------------- Objective Prototype --------------


class Objective:
    """Training objective"""

    def _init(self, module):
        """
        Parameters
        ----------
        module : fnn.model.modules.Module
            module to optimize
        """
        assert isinstance(module, self.dtype)
        self.module = module

    @property
    def dtype(self):
        """
        Returns
        -------
        type(fnn.model.modules.Module)
            module type
        """
        raise NotImplementedError()

    def objective(self, training=True, **data):
        """
        Parameters
        ----------
        training : bool
            training or validation
        **data
            training/validation data

        Returns
        -------
        float
            objective value
        """
        raise NotImplementedError()


# -------------- Objective Types --------------


class Architecture(Objective):
    """Architecture Objective"""

    def __init__(self, sample_stream=True):
        """
        Parameters
        ----------
        sample_stream : bool
            sample stream during training
        """
        self.sample_stream = bool(sample_stream)

    @property
    def dtype(self):
        return architectures.Architecture

    def objective(self, targets, stimuli, perspectives=None, modulations=None, training=True):
        """
        Parameters
        ----------
        stimulus : Iterable[ np.ndarray ]
            either singular or batch
        perspective : Iterable[ np.ndarray ] | None
            either singular or batch
        modulations : Iterable[ np.ndarray ] | None
            either singular or batch
        training : bool
            training or validation

        Returns
        -------
        float
            objective value
        """
        if training and self.sample_stream:
            stream = torch.randint(0, self.module.streams, (1,)).item()
        else:
            stream = None

        losses = self.module.generate_losses(
            targets=targets,
            stimuli=stimuli,
            perspectives=perspectives,
            modulations=modulations,
            training=training,
        )
        losses = list(losses)

        if training:
            objective = torch.stack(losses).mean()
            objective.backward()
        else:
            objective = np.stack(losses).mean()

        return objective.item()
