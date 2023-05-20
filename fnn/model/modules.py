from torch import nn
from itertools import chain


class Module(nn.Module):
    """Module"""

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

    def _reset(self):
        return

    def _restart(self):
        return

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

    def dropout(self, p: float = 0):
        from .elements import Dropout

        def fn(module):
            if isinstance(module, Dropout):
                module.p = p

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
    """Sequential Module"""

    pass


class ModuleList(nn.ModuleList, Module):
    """Module List"""

    pass
