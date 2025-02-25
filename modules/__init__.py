import bound_propagation as bp
import torch
import types

from .arithmetic import ScalarMult, ScalarAdd, Sum, SqNorm
from .linear import BoundLinear, Linear
from .sigmoid import BoundSigmoid
from .saturation import BoundClamp
from .sin import BoundSin
from .sequential import BoundSequential

linear_factory = bp.BoundModelFactory()
linear_factory.register(torch.nn.Sigmoid, BoundSigmoid)
linear_factory.register(Linear, BoundLinear)
linear_factory.register(bp.Clamp, BoundClamp)
linear_factory.register(bp.Sin, BoundSin)
linear_factory.register(torch.nn.Sequential, BoundSequential)


def overwrite_build(old_build):
    def new_build(self, module):
        if isinstance(module, (torch.nn.Sequential, Linear, torch.nn.Sigmoid, bp.Clamp, bp.Sin)):
            return old_build(module)
        else:
            raise NotImplementedError('strict linear bound propagation not supported for module type')
    return new_build

linear_factory.build = types.MethodType(overwrite_build(linear_factory.build), linear_factory)

__all__ = [
    'ScalarMult',
    'ScalarAdd',
    'Sum',
    'SqNorm',
    'linear_factory',
    'BoundSin'
]
