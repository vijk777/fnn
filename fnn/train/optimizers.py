import torch


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

    def optimize(self, data, step):
        """
        Parameters
        ----------
        data : Callable[[bool, int], Iterable[dict]]
            function that yields data batches
        step : Callable[[bool, int, ...], 1D array]
            function that returns loss values and performs gradient descent step if training=True
        """
        while self.scheduler.step():

            info = self.scheduler(**self.hyperparameters)
            info["cycle_epoch"] = self.scheduler.epoch
            info["global_epoch"] = self.scheduler.size * self.scheduler.cycle + self.scheduler.epoch

            for training, desc in [[True, "training"], [False, "validation"]]:

                losses = []

                for batch in data(training=training, epoch=info["global_epoch"]):

                    loss = step(training=training, epoch=info["global_epoch"], **batch)
                    losses.append(loss)

                    self.step()

                if losses:
                    info[f"{desc}_loss"] = np.stack(losses, axis=0).mean(axis=0)

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
