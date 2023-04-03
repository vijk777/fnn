import torch
from torch import nn
from itertools import chain


class Module(nn.Module):
    def named_containers(self):
        for name, module in self.named_children():
            if isinstance(module, Module):
                yield name, module

    def containers(self):
        for _, module in self.named_containers():
            yield module

    def reset(self):
        for module in self.containers():
            module.reset()
        self._reset()

    def _reset(self):
        pass

    def restart(self):
        for module in self.containers():
            module.restart()
        self._restart()

    def _restart(self):
        pass

    @property
    def device(self):
        try:
            params = chain(self.parameters(), self.buffers())
            device = next(params).device
        except:
            device = None
        return device

    @property
    def frozen(self):
        return getattr(self, "_frozen", False)

    def freeze(self, mode=True):
        self._frozen = bool(mode)
        self.requires_grad_(not self._frozen)
        self.train(not self._frozen)

        for module in self.containers():
            module.freeze(self._frozen)

        return self

    def requires_grad_(self, requires_grad=True):
        return super().requires_grad_(requires_grad and not self.frozen)

    def train(self, mode=True):
        return super().train(mode and not self.frozen)

    def special_param_groups(self, **kwargs):
        return []

    def param_groups(self, **kwargs):
        param_groups = []
        p = set(self.parameters())
        n = len(p)

        # collect parameters in children
        for module in self.containers():
            for group in module.param_groups(**kwargs):
                param_groups.append(group)
                p -= set(group["params"])
                n -= len(group["params"])

        # collect special parameters
        for group in self.special_param_groups(**kwargs):
            param_groups.append(group)
            p -= set(group["params"])
            n -= len(group["params"])

        # collect remaining parameters
        if p:
            param_groups.append({"params": list(p), **kwargs})
            n -= len(p)

        # assert all parameters were collected once
        assert n == 0

        return param_groups

    def _param_norm_dims(self):
        return dict()

    def param_norm_dims(self):
        # parameters with dedicated norm dimensions
        param_norm_dim = dict(self._param_norm_dims())

        # iterate over containers
        for module in self.containers():
            _param_norm_dim = module.param_norm_dims()

            # ensure parameters are non-overlapping
            assert set(param_norm_dim.keys()).isdisjoint(set(_param_norm_dim.keys()))

            # collect parameter norm dims
            param_norm_dim.update(_param_norm_dim)

        return param_norm_dim


class ModuleList(nn.ModuleList, Module):
    pass


class Sequential(nn.Sequential, Module):
    pass
