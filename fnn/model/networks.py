import torch
import numpy as np
from itertools import repeat
from .modules import Module


# -------------- Network Base --------------


class Network(Module):
    """Network"""

    def _init(self, stimuli, perspectives, modulations, units, streams):
        """
        Parameters
        ----------
        stimuli : int
            stimulus channels (C)
        perspectives : int
            perspective features (P)
        modulations : int
            modulation features (M)
        units : int
            number of units (U)
        streams : int
            number of streams (S)
        """
        raise NotImplementedError()

    def forward(self, stimulus, perspective, modulation, stream=None):
        """
        Parameters
        ----------
        stimulus : ND Tensor
            [N, C, ...] -- stimulus frame
        perspective : ND Tensor
            [N, P] -- perspective frame
        modulations : ND Tensor
            [N, M] -- modulation frame
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        2D Tensor
            [N, U] -- response frame
        """
        raise NotImplementedError()

    def loss(self, stimulus, perspective, modulation, unit, stream=None):
        """
        Parameters
        ----------
        stimulus : ND Tensor
            [N, C, ...] -- stimulus frame
        perspective : ND Tensor
            [N, P] -- perspective frame
        modulations : ND Tensor
            [N, M] -- modulation frame
        unit : ND Tensor
            [N, U] --  unit frame
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, U] -- loss frame
        """
        raise NotImplementedError()

    def generate_response(self, stimuli, perspectives=None, modulations=None, training=False):
        """
        Parameters
        ----------
        stimulus : Iterable[ND array]
            either singular or batch
        perspective : Iterable[ND array] | None
            either singular or batch
        modulations : Iterable[ND array] | None
            either singular or batch
        training : bool
            training or inference

        Yields
        ------
        either 1D array
            [U] (singular input, training=False)
        or 1D Tensor
            [U] (singular input, training=True)
        or 2D array
            [N, U] (batch input, training=False)
        or 2D Tensor
            [N, U] (batch input, training=True)
        """
        raise NotImplementedError()

    def generate_loss(self, units, stimuli, perspectives=None, modulations=None, stream=None, training=False):
        """
        Parameters
        ----------
        units : Iterable[ND array]
            either singular or batch
        stimulus : Iterable[ND array]
            either singular or batch
        perspective : Iterable[ND array] | None
            either singular or batch
        modulations : Iterable[ND array] | None
            either singular or batch
        stream : int | None
            specific stream (int) or all streams (None)
        training : bool
            training or inference

        Yields
        ------
        either 1D array
            [U] (singular input, training=False)
        or 1D Tensor
            [U] (singular input, training=True)
        or 2D array
            [N, U] (batch input, training=False)
        or 2D Tensor
            [N, U] (batch input, training=True)
        """
        raise NotImplementedError()

    def parallel_groups(self, group_size=1):
        """
        Parameters
        ----------
        group_size : int
            parallel group size

        Yields
        ------
        fnn.train.parallel.ParameterGroup
        """
        raise NotImplementedError()


# -------------- Network Types --------------


class Visual(Network):
    """Visual Network"""

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
            streams=self.streams,
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

    def _raw(self, stimulus, perspective, modulation, stream=None, periphery="dark"):
        """
        Parameters
        ----------
        stimulus : 4D Tensor
            [N, C, H, W] -- stimulus frame
        perspective : 2D Tensor
            [N, P] -- perspective frame
        modulations : 2D Tensor
            [N, M] -- modulation frame
        stream : int | None
            specific stream (int) or all streams (None)
        periphery : str
            "dark" | "extend"

        Returns
        -------
        3D Tensor
            [N, U, R] -- raw output
        """
        if periphery == "dark":
            perspective = self.perspective(
                stimulus=stimulus,
                perspective=perspective,
                pad_mode="zeros",
            )
        elif periphery == "extend":
            perspective = self.perspective(
                stimulus=stimulus,
                perspective=perspective,
                pad_mode="replicate",
            )
        else:
            raise ValueError(f"Invalid periphery -- {periphery}")

        if stream is None:
            perspective = perspective.repeat(1, self.streams, 1, 1)
            modulation = modulation.repeat(1, self.streams)

        modulation = self.modulation(
            modulation=modulation,
            stream=stream,
        )
        core = self.core(
            perspective=perspective,
            modulation=modulation,
            stream=stream,
        )
        readout = self.readout(
            core=core,
            stream=stream,
        )
        if stream is None:
            return self.reduce(readout)
        else:
            return readout

    def forward(self, stimulus, perspective, modulation, stream=None, periphery="dark"):
        """
        Parameters
        ----------
        stimulus : 4D Tensor
            [N, C, H, W] -- stimulus frame
        perspective : 2D Tensor
            [N, P] -- perspective frame
        modulations : 2D Tensor
            [N, M] -- modulation frame
        stream : int | None
            specific stream (int) or all streams (None)
        periphery : str
            "dark" | "extend"

        Returns
        -------
        2D Tensor
            [N, U] -- response frame
        """
        r = self._raw(
            stimulus=stimulus,
            perspective=perspective,
            modulation=modulation,
            stream=stream,
            periphery=periphery,
        )
        return self.unit(readout=r)

    def loss(self, stimulus, perspective, modulation, unit, stream=None):
        """
        Parameters
        ----------
        stimulus : 4D Tensor
            [N, C, H, W] -- stimulus frame
        perspective : 2D Tensor
            [N, P] -- perspective frame
        modulations : 2D Tensor
            [N, M] -- modulation frame
        unit : 2D Tensor
            [N, U] --  unit frame
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        2D Tensor
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

        Returns
        -------
        4D Tensor
            [N, C, H, W]
        2D Tensor
            [N, P]
        2D Tensor
            [N, M]
        bool
            squeeze response batch dim
        """
        assert stimulus.dtype == np.uint8
        stimulus = stimulus / 255

        if stimulus.ndim == 2:
            stimulus = stimulus[None, None]
            squeeze = True
            N = 1
        elif stimulus.ndim == 3:
            stimulus = np.einsum("H W C -> C H W", stimulus)[None]
            squeeze = True
            N = 1
        else:
            stimulus = np.einsum("N H W C -> N C H W", stimulus)
            squeeze = False
            N = stimulus.shape[0]

        if perspective is None:
            perspective = np.zeros([1, self.perspectives])
        elif perspective.ndim == 1:
            perspective = perspective[None]
        else:
            squeeze = False
            N = max(N, perspective.shape[0])

        if modulation is None:
            modulation = np.zeros([1, self.modulations])
        elif modulation.ndim == 1:
            modulation = modulation[None]
        else:
            squeeze = False
            N = max(N, modulation.shape[0])

        device = self.device
        tensor = lambda x: torch.tensor(x, dtype=torch.float, device=device)

        stimulus = tensor(stimulus).expand(N, -1, -1, -1)
        perspective = tensor(perspective).expand(N, -1)
        modulation = tensor(modulation).expand(N, -1)

        return stimulus, perspective, modulation, squeeze

    def generate_response(self, stimuli, perspectives=None, modulations=None, training=False):
        """
        Parameters
        ----------
        stimulus : Iterable[2D|3D|4D array]
            T x [H, W] | T x [H, W, C] | T x [N, H, W, C] --- dtype=uint8
        perspective : Iterable[1D|2D array] | None
            T x [P] | T x [N, P] --- dtype=float
        modulations : Iterable[1D|2D array] | None
            T x [M] | T x [N, M] --- dtype=float
        training : bool
            training or inference

        Yields
        ------
        either 1D array
            [U] (singular input, training=False)
        or 1D Tensor
            [U] (singular input, training=True)
        or 2D array
            [N, U] (batch input, training=False)
        or 2D Tensor
            [N, U] (batch input, training=True)
        """
        self.reset()

        if perspectives is None:
            perspectives = repeat(None)

        if modulations is None:
            modulations = repeat(None)

        with self.train_context(training):

            for stimulus, perspective, modulation in zip(stimuli, perspectives, modulations):

                *tensors, squeeze = self.to_tensor(stimulus, perspective, modulation)

                response = self(*tensors)
                if squeeze:
                    response = response.squeeze(0)

                if training:
                    yield response
                else:
                    yield response.cpu().numpy()

    def generate_loss(self, units, stimuli, perspectives=None, modulations=None, stream=None, training=False):
        """
        Parameters
        ----------
        units : Iterable[1D|2D array]
            T x [U] | T x [N, U] -- dtype=float
        stimulus : Iterable[2D|3D|4D array]
            T x [H, W] | T x [H, W, C] | T x [N, H, W, C] --- dtype=uint8
        perspective : Iterable[1D|2D array] | None
            T x [P] | T x [N, P] --- dtype=float
        modulations : Iterable[1D|2D array] | None
            T x [M] | T x [N, M] --- dtype=float
        stream : int | None
            specific stream (int) or all streams (None)
        training : bool
            training or inference

        Yields
        ------
        either 1D array
            [U] (singular input, training=False)
        or 1D Tensor
            [U] (singular input, training=True)
        or 2D array
            [N, U] (batch input, training=False)
        or 2D Tensor
            [N, U] (batch input, training=True)
        """
        self.reset()
        device = self.device

        if perspectives is None:
            perspectives = repeat(None)

        if modulations is None:
            modulations = repeat(None)

        with self.train_context(training):

            for unit, stimulus, perspective, modulation in zip(units, stimuli, perspectives, modulations):

                unit = torch.tensor(unit, dtype=torch.float, device=device)
                if unit.ndim == 1:
                    unit = unit[None]
                    squeeze = True
                else:
                    squeeze = False

                *tensors, _squeeze = self.to_tensor(stimulus, perspective, modulation)
                assert squeeze == _squeeze

                loss = self.loss(*tensors, unit=unit, stream=stream)
                if squeeze:
                    loss = loss.squeeze(0)

                if training:
                    yield loss
                else:
                    yield loss.cpu().numpy()

    def parallel_groups(self, group_size=1):
        """
        Parameters
        ----------
        group_size : int
            parallel group size

        Yields
        ------
        fnn.train.parallel.ParameterGroup
        """
        from torch.distributed import is_initialized, get_world_size, get_rank, new_group
        from fnn.train.parallel import ParameterGroup

        if not is_initialized():
            assert group_size == 1
            return

        size = get_world_size()
        assert size % group_size == 0

        if not size > 1:
            return

        if not self.core.frozen:
            ranks = np.arange(size)
            yield ParameterGroup(
                parameters=self.core.named_parameters(),
                group=new_group(ranks),
            )

        if group_size > 1:
            params = dict()

            for name in ["perspective", "modulation", "readout", "reduce", "unit"]:
                module = getattr(self, name)
                if not module.frozen:
                    params = dict(params, **{f"{name}.{k}": v for k, v in module.named_parameters()})

            if params:
                ranks = get_rank() // group_size * group_size + np.arange(group_size)
                yield ParameterGroup(
                    parameters=params,
                    group=new_group(ranks),
                )
