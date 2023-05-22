import torch
import numpy as np
from itertools import repeat
from contextlib import nullcontext
from .modules import Module


# -------------- Architecture Prototype --------------


class Architecture(Module):
    """Architecture Module"""

    def _init(self, stimuli, perspectives, modulations, units, streams):
        """
        Parameters
        ----------
        stimuli : int
            stimulus channels
        perspectives : int
            perspective features
        modulations : int
            modulation features
        units : int
            number of units
        streams : int
            number of streams
        """
        raise NotImplementedError()

    def generate_responses(self, stimuli, perspectives=None, modulations=None, training=False):
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
            training or inference

        Yields
        ------
        either 1D np.ndarry
            [U] (singular input, training=False)
        or 1D Tensor
            [U] (singular input, training=True)
        or 2D np.ndarry
            [N, U] (batch input, training=False)
        or 2D Tensor
            [N, U] (batch input, training=True)
        """
        raise NotImplementedError()

    def generate_losses(self, units, stimuli, perspectives=None, modulations=None, stream=None, training=False):
        """
        Parameters
        ----------
        units : Iterable[ np.ndarray ]
            either singular or batch
        stimulus : Iterable[ np.ndarray ]
            either singular or batch
        perspective : Iterable[ np.ndarray ] | None
            either singular or batch
        modulations : Iterable[ np.ndarray ] | None
            either singular or batch
        stream : int | None
            specific stream | all streams
        training : bool
            training or inference

        Yields
        ------
        either 1D np.ndarry
            [U] (singular input, training=False)
        or 1D Tensor
            [U] (singular input, training=True)
        or 2D np.ndarry
            [N, U] (batch input, training=False)
        or 2D Tensor
            [N, U] (batch input, training=True)
        """
        raise NotImplementedError()

    def parallel_groups(self, instances=1):
        """
        Parameters
        ----------
        instances : int
            training instances

        Yields
        ------
        fnn.train.parallel.ParameterGroup
        """
        raise NotImplementedError()


# -------------- Architecture Types --------------


class Visual(Architecture):
    """Visual Cortex"""

    def __init__(self, core, perspective, modulation, readout, reduce, unit):
        """
        Parameters
        ----------
        core : fnn.model.cores.Core
            core model
        perspective : fnn.model.perspectives.Perspective
            perspective model
        modulation : fnn.model.modulations.Modulation
            modulation model
        readout : fnn.model.readouts.Readout
            readout model
        reduce : fnn.model.reductions.Reduce
            stream reduction
        unit : fnn.model.units.Unit
            neuronal unit
        """
        super().__init__()
        self.core = core
        self.perspective = perspective
        self.modulation = modulation
        self.readout = readout
        self.reduce = reduce
        self.unit = unit

    def _init(self, stimuli, perspectives, modulations, units, streams):
        """
        Parameters
        ----------
        stimuli : int
            stimulus channels
        perspectives : int
            perspective features
        modulations : int
            modulation features
        units : int
            number of units
        streams : int
            number of streams
        """
        self.stimuli = int(stimuli)
        self.perspectives = int(perspectives)
        self.modulations = int(modulations)
        self.units = int(units)
        self.streams = int(streams)

        self.perspective._init(
            stimuli=stimuli,
            perspectives=self.perspectives,
        )
        self.modulation._init(
            modulations=self.modulations,
        )
        self.core._init(
            perspectives=self.perspective.channels,
            modulations=self.modulation.features,
            streams=self.streams,
        )
        self.readout._init(
            cores=self.core.channels,
            readouts=self.unit.readouts,
            units=self.units,
            streams=self.streams,
        )
        self.reduce._init(
            dim=[1],
            keepdim=False,
        )

    def _raw(self, stimulus, perspective, modulation, stream=None):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, C, H, W]
        perspective : Tensor
            [N, P]
        modulations : Tensor
            [N, M]
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            [N, U, R] -- raw output
        """
        p = self.perspective(
            stimulus=stimulus,
            perspective=perspective,
        )
        m = self.modulation(
            modulation=modulation,
        )
        c = self.core(
            perspective=p,
            modulation=m,
            stream=stream,
        )
        r = self.readout(
            core=c,
            stream=stream,
        )
        if stream is None:
            return self.reduce(r)
        else:
            return r

    def forward(self, stimulus, perspective, modulation, stream=None):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, C, H, W]
        perspective : Tensor
            [N, P]
        modulations : Tensor
            [N, M]
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            [N, U] -- response frame
        """
        r = self._raw(
            stimulus=stimulus,
            perspective=perspective,
            modulation=modulation,
            stream=stream,
        )
        return self.unit(readout=r)

    def loss(self, stimulus, perspective, modulation, unit, stream=None):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, C, H, W]
        perspective : Tensor
            [N, P]
        modulations : Tensor
            [N, M]
        unit : Tensor
            [N, U]
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            [N, U] -- loss frame
        """
        r = self._raw(
            stimulus=stimulus,
            perspective=perspective,
            modulation=modulation,
            stream=stream,
        )
        return self.unit.loss(readout=r, unit=unit)

    def to_tensor(self, stimulus, perspective=None, modulation=None):
        """
        Parameters
        ----------
        stimulus : 2D|3D|4D array
            [H, W] | [H, W, C] | [N, H, W, C]
        perspective : 1D|2D array | None
            [P] | [N, P]
        modulations : 1D|2D array | None
            [M] | [N, M]
        """
        assert stimulus.dtype == np.uint8
        stimulus = stimulus / 255

        if stimulus.ndim == 2:
            stimulus = stimulus[None, None]
            squeeze = True
        elif stimulus.ndim == 3:
            stimulus = np.einsum("H W C -> C H W", stimulus)[None]
            squeeze = True
        else:
            stimulus = np.einsum("N H W C -> N C H W", stimulus)
            squeeze = False

        if perspective is None:
            perspective = np.zeros([1, self.perspectives])
        elif perspective.ndim == 1:
            assert squeeze
            perspective = perspective[None]
        else:
            assert not squeeze

        if modulation is None:
            modulation = np.zeros([1, self.modulations])
        elif modulation.ndim == 1:
            assert squeeze
            modulation = modulation[None]
        else:
            assert not squeeze

        device = self.device
        tensor = lambda x: torch.tensor(x, dtype=torch.float, device=device)

        return tensor(stimulus), tensor(perspective), tensor(modulation), squeeze

    def generate_responses(self, stimuli, perspectives=None, modulations=None, training=False):
        """
        Parameters
        ----------
        stimulus : Iterable[ 2D|3D|4D np.ndarray ]
            [H, W] | [H, W, C] | [N, H, W, C] x T --- dtype=uint8
        perspective : Iterable[ 1D|2D np.ndarray ] | None
            [P] | [N, P] x T --- dtype=float
        modulations : Iterable[ 1D|2D np.ndarray ] | None
            [M] | [N, M] x T --- dtype=float
        training : bool
            training or inference

        Yields
        ------
        either 1D np.ndarry
            [U] (singular input, training=False)
        or 1D Tensor
            [U] (singular input, training=True)
        or 2D np.ndarry
            [N, U] (batch input, training=False)
        or 2D Tensor
            [N, U] (batch input, training=True)
        """
        self.reset()

        if perspectives is None:
            perspectives = repeat(None)

        if modulations is None:
            modulations = repeat(None)

        _training = self.training
        if training:
            context = nullcontext
            self.train(True)
        else:
            context = torch.inference_mode
            self.train(False)

        with context():

            for stimulus, perspective, modulation in zip(stimuli, perspectives, modulations):

                *tensors, squeeze = self.to_tensor(stimulus, perspective, modulation)
                prediction = self(*tensors)

                if squeeze:
                    prediction = prediction.squeeze(0)

                if training:
                    yield prediction
                else:
                    yield prediction.cpu().numpy()

        self.train(_training)

    def generate_losses(self, units, stimuli, perspectives=None, modulations=None, stream=None, training=False):
        """
        Parameters
        ----------
        units : Iterable[ 1D|2D np.narray ]
            [U] | [N, U] x T -- dtype=float
        stimulus : Iterable[ 2D|3D|4D np.ndarray ]
            [H, W] | [H, W, C] | [N, H, W, C] x T --- dtype=uint8
        perspective : Iterable[ 1D|2D np.ndarray] | None
            [P] | [N, P] x T --- dtype=float
        modulations : Iterable[ 1D|2D np.ndarray ] | None
            [M] | [N, M] x T --- dtype=float
        stream : int | None
            specific stream | all streams
        training : bool
            training or inference

        Yields
        ------
        either 1D np.ndarry
            [U] (singular input, training=False)
        or 1D Tensor
            [U] (singular input, training=True)
        or 2D np.ndarry
            [N, U] (batch input, training=False)
        or 2D Tensor
            [N, U] (batch input, training=True)
        """
        self.reset()

        if perspectives is None:
            perspectives = repeat(None)

        if modulations is None:
            modulations = repeat(None)

        _training = self.training
        if training:
            context = nullcontext
            self.train(True)
        else:
            context = torch.inference_mode
            self.train(False)

        with context():

            for unit, stimulus, perspective, modulation in zip(units, stimuli, perspectives, modulations):

                *tensors, squeeze = self.to_tensor(stimulus, perspective, modulation)

                unit = torch.tensor(unit, dtype=torch.float, device=self.device)
                if unit.ndim == 1:
                    assert squeeze
                    unit = unit[None]
                else:
                    assert not squeeze

                loss = self.loss(*tensors, unit=unit, stream=stream)

                if squeeze:
                    loss = loss.squeeze(0)

                if training:
                    yield loss
                else:
                    yield loss.cpu().numpy()

        self.train(_training)

    def parallel_groups(self, instances=1):
        """
        Parameters
        ----------
        instances : int
            training instances

        Yields
        ------
        fnn.train.parallel.ParameterGroup
        """
        from torch.distributed import is_initialized, get_world_size, get_rank, new_group
        from fnn.train.parallel import ParameterGroup

        if not is_initialized():
            assert instances == 1
            return

        size = get_world_size()
        assert size % instances == 0

        if not size > 1:
            return

        if not self.core.frozen:
            ranks = np.arange(size)
            yield ParameterGroup(
                parameters=self.core.named_parameters(),
                group=new_group(ranks),
            )

        if instances > 1:
            params = dict()

            for name in ["perspective", "modulation", "readout", "reduce", "unit"]:
                module = getattr(self, name)
                if not module.frozen:
                    params = dict(params, **{f"{name}.{k}": v for k, v in module.named_parameters()})

            if params:
                ranks = get_rank() // instances * instances + np.arange(instances)
                yield ParameterGroup(
                    parameters=params,
                    group=new_group(ranks),
                )
