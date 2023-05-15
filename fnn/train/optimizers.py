import torch
import numpy as np
from .parallel import ParameterGroup


# -------------- Optimizer Prototype --------------


class Optimizer:
    """Module Optimizer"""

    def _init(self, module, scheduler):
        """
        Parameters
        ----------
        module : fnn.model.modules.Module
            module to optimize
        scheduler : fnn.train.schedulers.Scheduler
            hyperparameter scheduler
        """
        self.module = module
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

    def step(self):
        """
        Perform a gradient descent step
        """
        raise NotImplementedError()

    def optimize(self, loader, objective, seed=42, parallel=None):
        """
        Parameters
        ----------
        loader : fnn.train.loaders.Loader
            data loader
        objective : fnn.train.objectives.Objective
            training objective
        seed : int
            random seed
        parallel : None | list[ tuple[str, torch.distributed.ProcessGroup] ]
            None or [(`component`, `process group`), ...]
        """
        objective._init(self.module)

        groups = []
        if parallel is not None:
            for component, group in parallel:

                c = getattr(self.module, component)
                if c.frozen:
                    continue

                g = ParameterGroup(parameters=c.named_parameters(), group=group)
                groups.append(g)

        while self.scheduler.step():

            info = self.scheduler(**self.hyperparameters)

            for g in groups:
                g.sync_params()

            for training, desc in [[True, "training"], [False, "validation"]]:

                objectives = []

                device = torch.cuda.current_device()
                print(device)  # TODO: REMOVE

                with torch.random.fork_rng([device]):

                    _seed = seed + self.scheduler.size * self.scheduler.cycle + self.scheduler.epoch
                    torch.manual_seed(_seed)

                    for data in loader(training=training):

                        o = objective(training=training, **data)

                        if not np.isfinite(o):
                            raise ValueError("Non-finite objective.")

                        if training:
                            for g in groups:
                                g.sync_grads()

                            self.step()

                        objectives.append(o)

                if objectives:
                    info[f"{desc}_objective"] = np.mean(objectives)

            yield info


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

    def _init(self, module, scheduler):
        """
        Parameters
        ----------
        module : fnn.model.modules.Module
            module to optimize
        scheduler : fnn.train.schedulers.Scheduler
            hyperparameter scheduler
        """
        super()._init(module, scheduler)
        self.param_groups = list(module.param_groups(**self.hyperparameters))
        self.norm_dims = dict(module.param_norm_dims())
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
    def step(self):
        """
        Perform a gradient descent step
        """
        for group in self.param_groups:

            hp = self.scheduler(**{k: group[k] for k in self.hyperparameters})

            lr = hp["lr"]
            momentum = hp["momentum"]
            nesterov = hp["nesterov"]
            decay = hp["decay"]
            clip = hp["clip"]
            eps = hp["eps"]

            for p in group["params"]:

                d_p = p.grad

                if d_p is None:
                    continue

                if clip < float("inf"):
                    norm_dim = self.norm_dims.get(p)
                    p_norm = p.norm(p=2, dim=norm_dim, keepdim=True)
                    d_p_norm = d_p.norm(p=2, dim=norm_dim, keepdim=True)

                    min_norm = eps * p.numel() ** 0.5
                    max_norm = (clip * p_norm).clamp(min=min_norm)

                    c = max_norm / torch.maximum(d_p_norm, max_norm)
                    d_p = d_p.mul(c)

                if decay > 0:
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
