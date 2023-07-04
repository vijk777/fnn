import numpy as np
import torch


# -------------- Objective Bases --------------


class Objective:
    """Objective"""

    def __call__(self, training=True, **data):
        """Performs an objective call

        Parameters
        ----------
        training : bool
            training or validation
        **data
            training or validation data
        """
        raise NotImplementedError()

    def step(self):
        """Perform an epoch step

        Returns
        -------
        dict
            epoch info
        """
        raise NotImplementedError()


class NetworkObjective(Objective):
    """Network Objective"""

    def _init(self, network):
        """
        Parameters
        ----------
        network : fnn.model.networks.Network
            network module
        """
        raise NotImplementedError()


class StimulusObjective(Objective):
    """Stimulus Objective"""

    def _init(self, network, stimulus):
        """
        Parameters
        ----------
        module : fnn.model.networks.Network
            network module
        stimulus : fnn.model.stimuli.Stimulus
            stimulus module
        """
        raise NotImplementedError()


# -------------- Objective Types --------------


class NetworkLoss(NetworkObjective):
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

    def _init(self, network):
        """
        Parameters
        ----------
        network : fnn.model.networks.Network
            network module
        """
        self.network = network
        self._training = []
        self._validation = []

    def __call__(self, units, stimuli, perspectives=None, modulations=None, training=True):
        """Performs an objective call

        Parameters
        ----------
        stimulus : Iterable[ND array]
            either singular or batch
        perspective : Iterable[ND array] | None
            either singular or batch
        modulations : Iterable[ND array] | None
            either singular or batch
        training : bool
            training or validation
        """
        if training and self.sample_stream:
            stream = torch.randint(0, self.network.streams, (1,)).item()
        else:
            stream = None

        losses = self.network.generate_loss(
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

        objective = objective.item()

        if not np.isfinite(objective):
            raise ValueError("Non-finite objective")

        if training:
            self._training.append(objective)
        else:
            self._validation.append(objective)

    def step(self):
        """Perform an epoch step

        Returns
        -------
        dict[str, float]
            epoch objectives
        """
        objectives = dict()

        if self._training:
            objectives["training_objective"] = np.mean(self._training)
            self._training.clear()

        if self._validation:
            objectives["validation_objective"] = np.mean(self._validation)
            self._validation.clear()

        return objectives


class Reconstruction(StimulusObjective):
    """Reconstruction"""

    def __init__(
        self,
        trial_perspectives,
        trial_modulations,
        trial_units,
        sample_trial=True,
        sample_stream=True,
        burnin_frames=0,
        penalty=0,
    ):
        """
        Parameters
        ----------
        trial_perspectives : 3D array
            [timepoints, trials, perspectives]
        trial_modulations : 3D array
            [timepoints, trials, modulations]
        trial_units : 3D array
            [timepoints, trials, units]
        sample_trial : bool
            sample trial during training
        sample_stream : bool
            sample stream during training
        burnin_frames : int
            number of initial frames to discard
        penalty : float
            stimulus penalty weight
        """
        assert burnin_frames >= 0
        assert penalty >= 0

        self.trial_perspectives = torch.tensor(trial_perspectives, dtype=torch.float)
        self.trial_modulations = torch.tensor(trial_modulations, dtype=torch.float)
        self.trial_units = torch.tensor(trial_units, dtype=torch.float)

        (self.timepoints,) = {
            self.trial_perspectives.shape[0],
            self.trial_modulations.shape[0],
            self.trial_units.shape[0],
        }

        (self.trials,) = {
            self.trial_perspectives.shape[1],
            self.trial_modulations.shape[1],
            self.trial_units.shape[1],
        }

        self.sample_trial = bool(sample_trial)
        self.sample_stream = bool(sample_stream)
        self.burnin_frames = int(burnin_frames)
        self.penalty = float(penalty)

    def _init(self, network, stimulus):
        """
        Parameters
        ----------
        module : fnn.model.networks.Network
            network module
        stimulus : fnn.model.stimuli.Stimulus
            stimulus module
        """
        self.network = network
        self.stimulus = stimulus

        self.trial_perspectives = self.trial_perspectives.to(device=self.network.device)
        self.trial_modulations = self.trial_modulations.to(device=self.network.device)
        self.trial_units = self.trial_units.to(device=self.network.device)

        self.log = dict(
            training_loss=[],
            training_penalty=[],
            validation_loss=[],
            validation_penalty=[],
        )

    def __call__(self, training=True):
        """Performs an objective call

        Parameters
        ----------
        training : bool
            training or validation
        """
        self.network.reset()

        if training and self.sample_trial:
            trial = torch.randint(0, self.trials, (1,)).item()
            select = lambda x: x[:, trial, None]
            expand = lambda x: x.unsqueeze(0)
        else:
            select = lambda x: x
            expand = lambda x: x.expand(self.trials, -1, -1, -1)

        if training and self.sample_stream:
            stream = torch.randint(0, self.network.streams, (1,)).item()
        else:
            stream = None

        with self.network.train_context(training):

            inputs = zip(self.stimulus(), perspectives, modulations, units)
            losses = []

            for timepoint, (stimulus, perspective, modulation, unit) in enumerate(inputs):

                loss = self.network.loss(
                    stimulus=expand(stimulus),
                    perspective=select(perspective),
                    modulation=select(modulation),
                    unit=select(unit),
                    stream=stream,
                )

                if timepoint >= self.burnin_frames:
                    losses.append(loss.mean())

            loss = torch.stack(losses).sum()
            penalty = self.stimulus.penalty()

            loss_item = loss.item()
            penalty_item = penalty.item()

            if not np.isfinite(loss_item):
                raise ValueError("Non-finite loss")

            if not np.isfinite(penalty_item):
                raise ValueError("Non-finite penalty")

            if training:
                stream = (loss + self.penalty * penalty).backward()

                self.log["training_loss"].append(loss_item)
                self.log["training_penalty"].append(penalty_item)

            else:
                self.log["validation_loss"].append(loss_item)
                self.log["validation_penalty"].append(penalty_item)

    def step(self):
        """Perform an epoch step

        Returns
        -------
        dict[str, float]
            epoch objectives
        """
        objectives = dict()

        for key, value in self.log.items():
            if value:
                objectives[key] = np.mean(value)
                value.clear()

        return objectives
