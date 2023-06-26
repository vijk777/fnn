import numpy as np
import torch


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
        self.module = module

    def __call__(self, training=True, **data):
        """
        Parameters
        ----------
        training : bool
            training or validation
        **data
            training or validation data

        Returns
        -------
        float
            training or validation objective
        """
        raise NotImplementedError()


# -------------- Objective Types --------------


class NetworkLoss(Objective):
    """Network Loss"""

    def __init__(self, sample_stream=True, burnin_frames=0):
        """
        Parameters
        ----------
        sample_stream : bool
            sample stream during training
        burnin_frames : int
            number of initial frames to discard
        """
        assert burnin_frames >= 0

        self.sample_stream = bool(sample_stream)
        self.burnin_frames = int(burnin_frames)

    def __call__(self, units, stimuli, perspectives=None, modulations=None, training=True):
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
            units=units,
            stimuli=stimuli,
            perspectives=perspectives,
            modulations=modulations,
            stream=stream,
            training=training,
        )
        losses = list(losses)[self.burnin_frames :]

        if training:
            objective = torch.stack(losses).mean()
            objective.backward()
        else:
            objective = np.stack(losses).mean()

        return objective.item()
