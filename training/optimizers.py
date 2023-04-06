import math
import torch

from networks.containers import Module


class SGD:
    def __init__(
        self,
        model: Module,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0,
        clip: float = math.inf,
        eps: float = 0.001,
    ):
        if lr < 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if clip < 0:
            raise ValueError("Invalid clip value: {}".format(clip))
        if eps <= 0:
            raise ValueError("Invalid eps value: {}".format(eps))

        self.defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            clip=clip,
            eps=eps,
        )
        self.param_groups = list(model.param_groups(**self.defaults))
        self.norm_dims = dict(model.param_norm_dims())

    @torch.no_grad()
    def step(self, lr_scale=1):
        """Perform a single optimization step

        Parameters
        ----------
        lr_scale : float
            multiplies the learning rate by this amount
        """
        assert lr_scale > 0

        for group in self.param_groups:

            lr = group["lr"] * lr_scale
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            clip = group["clip"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad

                if clip < math.inf:
                    norm_dim = self.norm_dims.get(p)
                    p_norm = p.norm(p=2, dim=norm_dim, keepdim=True)
                    d_p_norm = d_p.norm(p=2, dim=norm_dim, keepdim=True)

                    max_norm = clip * p_norm.clamp(eps, None)
                    clip_coef = max_norm / torch.maximum(d_p_norm, max_norm)
                    d_p = d_p.mul(clip_coef)

                if weight_decay > 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum > 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p)
                        d_p = d_p.add(buf, alpha=momentum)

                p.add_(d_p, alpha=-lr)
