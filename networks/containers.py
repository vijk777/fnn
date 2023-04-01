import torch
from torch import nn


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("epoch", torch.tensor(0, dtype=torch.long))

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

    def step_epoch(self):
        for module in self.containers():
            module.step_epoch()
        self.epoch = self.epoch + 1

    @property
    def device(self):
        try:
            device = next(self.parameters()).device
        except:
            device = None
        return device

    @property
    def frozen(self):
        return getattr(self, "_frozen", False)

    def freeze(self, mode=True):
        if not isinstance(mode, bool):
            raise TypeError("mode must be boolean")

        self._frozen = mode
        self.requires_grad_(not mode)
        self.train(not mode)

        for module in self.containers():
            module.freeze(mode)

        return self

    def requires_grad_(self, requires_grad=True):
        return super().requires_grad_(requires_grad and not self.frozen)

    def train(self, mode=True):
        return super().train(mode and not self.frozen)

    def special_param_groups(self, **kwargs):
        return []

    def param_groups(self, **kwargs):
        # collect special parameters
        param_groups = self.special_param_groups(**kwargs)
        special_parameters = set()
        for group in param_groups:
            special_parameters |= set(group["params"])

        # collect other parameters
        parameters = set()
        for module in self.containers():
            _param_groups = module.param_groups(**kwargs)
            for group in _param_groups:
                parameters |= set(group["params"])
            param_groups += _param_groups

        # ensure special parameters and other parameters are disjoint
        assert special_parameters.isdisjoint(parameters)

        # collect remaining parameters in modeuls
        remaining_parameters = set(self.parameters(recurse=True)) - special_parameters - parameters
        if remaining_parameters:
            remaining_parameters_list = [p for p in self.parameters(recurse=True) if p in remaining_parameters]
            if remaining_parameters_list:
                param_groups += [{"params": remaining_parameters_list, **kwargs}]

        return param_groups

    def _param_norm_dim(self):
        return dict()

    def param_norm_dim(self):
        # parameters with dedicated norm dimensions
        param_norm_dim = dict(self._param_norm_dim())

        # iterate over containers
        for module in self.containers():
            _param_norm_dim = module.param_norm_dim()

            # ensure parameters are non-overlapping
            assert set(param_norm_dim.keys()).isdisjoint(set(_param_norm_dim.keys()))

            # collect parameter norm dims
            param_norm_dim.update(_param_norm_dim)

        return param_norm_dim


class ModuleList(nn.ModuleList, Module):
    pass


class Sequential(nn.Sequential, Module):
    pass
