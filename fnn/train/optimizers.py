import torch


# -------------- Optimizer Prototype --------------


class Optimizer:
    """Module Optimizer"""

    def _init(self, module):
        """
        Parameters
        ----------
        module : fnn.model.modules.Module
            module to optimize
        """
        self.module = module
        self._initialized = True

    @property
    def initialized(self):
        """
        Returns
        -------
        bool
            whether optimizer has been initialized
        """
        return getattr(self, "_initialized", False)

    @property
    def hyperparameters(self):
        """
        Returns
        -------
        dict
            dictionary of hyperparameters
        """
        raise NotImplementedError()

    def step(self, scheduler):
        """Perform a single optimization step

        Parameters
        ----------
        scheduler : fnn.train.schedulers.Scheduler
            hyperparameter scheduler
        """
        raise NotImplementedError()


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
        assert decay > 0
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

    def _init(self, module):
        """
        Parameters
        ----------
        module : fnn.model.modules.Module
            module to optimize
        """
        super()._init(module)
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
    def step(self, scheduler):
        """Perform a single optimization step

        Parameters
        ----------
        scheduler : fnn.train.schedulers.Scheduler
            hyperparameter scheduler
        """
        for group in self.param_groups:

            hp = scheduler(**{k: group[k] for k in self.hyperparameters})

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
