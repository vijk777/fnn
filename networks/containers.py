from torch import nn
from itertools import chain


class Module(nn.Module):
    def _iterate(self, fn, memo=None):
        if memo is None:
            memo = set()

        for module in filter(lambda x: isinstance(x, Module), self.children()):

            if module in memo:
                continue
            else:
                memo.add(module)

            yield from module._iterate(fn, memo)

        vals = fn(self)
        if vals is None:
            return
        else:
            yield from vals

    def _param_groups(self, **kwargs):
        return
        yield

    def _param_norm_dims(self):
        return
        yield

    def _reset(self):
        return

    def _restart(self):
        return

    def param_groups(self, **kwargs):
        collected = set()

        def add_group(group):
            params = set(group["params"])

            assert len(params) == len(group["params"])
            assert params.isdisjoint(collected)

            collected.update(params)

        def fn(module):
            for group in module._param_groups(**kwargs):
                add_group(group)
                yield group

            params = set(module.parameters()) - collected
            if params:
                group = {"params": list(params), **kwargs}
                add_group(group)
                yield group

        yield from self._iterate(fn)

        assert collected == set(self.parameters())

    def param_norm_dims(self):
        collected = set()

        def fn(module):
            for param, norm_dim in module._param_norm_dims():

                assert param not in collected
                collected.update({param})

                yield param, norm_dim

        yield from self._iterate(fn)

    def reset(self):
        def fn(module):
            module._reset()

        all(self._iterate(fn))
        return self

    def restart(self):
        def fn(module):
            module._restart()

        all(self._iterate(fn))
        return self

    def freeze(self, mode: bool = True):
        def fn(module):
            module._frozen = bool(mode)
            module.requires_grad_(not mode)
            module.train(not mode)

        all(self._iterate(fn))
        return self

    def requires_grad_(self, requires_grad: bool = True):
        return super().requires_grad_(requires_grad and not self.frozen)

    def train(self, mode: bool = True):
        return super().train(mode and not self.frozen)

    @property
    def frozen(self):
        return getattr(self, "_frozen", False)

    @property
    def device(self):
        x = next(chain(self.parameters(), self.buffers(), [None]))
        if x is not None:
            return x.device


class Sequential(nn.Sequential, Module):
    pass


class ModuleList(nn.ModuleList, Module):
    pass
