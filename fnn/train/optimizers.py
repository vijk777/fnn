import torch
import numpy as np
from .parallel import ParameterGroup


# -------------- Optimizer Prototype --------------


class Optimizer:
    """Module Optimizer"""

    def _init(self, scheduler):
        """
        Parameters
        ----------
        module : fnn.model.modules.Module
            module to optimize
        scheduler : fnn.train.schedulers.Scheduler
            hyperparameter scheduler
        """
        self.scheduler = scheduler

    @property
    def hyperparameters(self):
        """
        Returns
        -------
        dict
            dictionary of hyperparameters
        """
        raise NotImplementedError()

    def step(self, parameters, **kwargs):
        """Perform a gradient descent step

        Parameters
        ----------
        parameters : list[fnn.model.parameters.Parameter]
            list of parameters
        **kwargs
            hyperparameters
        """
        raise NotImplementedError()

    def optimize(self, loader, objective, parameters, groups=None, seed=42):
        """
        Parameters
        ----------
        loader : fnn.train.loaders.Loader
            data loader
        objective : fnn.train.objectives.Objective
            training objective
        parameters : list[fnn.model.parameters.Parameter]
            list of parameters
        groups : None | list[fnn.train.parallel.ParameterGroup]
            None | list of parameter groups
        seed : int
            random seed

        Yields
        ------
        int
            epoch number
        dict
            optimization info (hyperparameters and objectives)
        """
        parameters = list(parameters)
        groups = [] if groups is None else list(groups)

        while self.scheduler.step():

            info = self.scheduler(**self.hyperparameters)

            for g in groups:
                g.sync_params()

            for training, desc in [[True, "training"], [False, "validation"]]:

                objectives = []

                devices = torch.cuda.device_count()
                devices = list(range(devices))
                with torch.random.fork_rng(devices):

                    _seed = seed + self.scheduler.seed
                    torch.manual_seed(_seed)

                    for data in loader(training=training):

                        o = objective(training=training, **data)

                        if not np.isfinite(o):
                            raise ValueError("Non-finite objective.")

                        if training:
                            for g in groups:
                                g.sync_grads()

                            self.step(parameters, **info)

                        objectives.append(o)

                if objectives:
                    info[f"{desc}_objective"] = np.mean(objectives)

            yield self.scheduler.epoch, info


# -------------- Optimizer Types --------------


class SgdClip(Optimizer):
    def __init__(self, lr=0.1, decay=0, momentum=0, nesterov=False, clip=float("inf"), eps=0.001):
        """
        Parameters
        ----------
        lr : float
            learning rate
        decay : float
            weight decay
        momentum : float
            momentum factor [0, 1)
        nesterov : bool
            enables nesterov momentum
        clip : float
            adaptive gradient clipping factor
        eps : float
            adaptive gradient clipping minimum
        """
        assert lr > 0
        assert decay >= 0
        assert 0 <= momentum < 1
        assert clip > 0
        assert eps > 0

        self._hyperparameters = dict(
            lr=float(lr),
            decay=float(decay),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            clip=float(clip),
            eps=float(eps),
        )

        self.momentums = dict()

    @property
    def hyperparameters(self):
        """
        Returns
        -------
        dict
            dictionary of hyperparameters
        """
        return self._hyperparameters

    @torch.no_grad()
    def step(self, parameters, lr, momentum, nesterov, decay, clip, eps):
        """Perform a gradient descent step

        Parameters
        ----------
        parameters : list[fnn.model.parameters.Parameter]
            list of parameters
        lr : float
            learning rate
        decay : float
            weight decay
        momentum : float
            momentum factor [0, 1)
        nesterov : bool
            enables nesterov momentum
        clip : float
            adaptive gradient clipping factor
        eps : float
            adaptive gradient clipping minimum
        """
        for p in parameters:

            d_p = p.grad

            if d_p is None:
                continue

            d_p = d_p.mul(p.scale)

            if clip < float("inf"):
                p_norm = p.norm(p=2, dim=p.norm_dim, keepdim=True)
                d_p_norm = d_p.norm(p=2, dim=p.norm_dim, keepdim=True)

                min_norm = eps * p.numel() ** 0.5
                max_norm = (clip * p_norm).clamp(min=min_norm)

                c = max_norm / torch.maximum(d_p_norm, max_norm)
                d_p = d_p.mul(c)

            if decay > 0 and p.decay:
                d_p = d_p.add(p, alpha=decay)

            if momentum > 0:
                m = self.momentums.get(p, None)

                if m is None:
                    m = self.momentums[p] = torch.clone(d_p)
                else:
                    m.mul_(momentum).add_(d_p)

                if nesterov:
                    d_p = d_p.add(m, alpha=momentum)
                else:
                    d_p = m

            p.add_(d_p, alpha=-lr)
            p.grad = None
