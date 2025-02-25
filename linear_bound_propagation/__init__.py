import bound_propagation as bp
import torch
import types

from .arithmetic import ScalarMult, ScalarAdd, Sum, SqNorm
from .linear import BoundLinear, Linear, Identity
from .sigmoid import BoundSigmoid
from .saturation import BoundClamp
from .sin import BoundSin
from .sequential import BoundSequential

factory = bp.BoundModelFactory()
factory.register(torch.nn.Sigmoid, BoundSigmoid)
factory.register(Linear, BoundLinear)
factory.register(bp.Clamp, BoundClamp)
factory.register(bp.Sin, BoundSin)
factory.register(torch.nn.Sequential, BoundSequential)


def overwrite_build(old_build):
    def new_build(self, module):
        if isinstance(module, (torch.nn.Sequential, Linear, torch.nn.Sigmoid, bp.Clamp, bp.Sin, bp.Parallel, bp.VectorAdd, bp.Sub)):
            return old_build(module)
        else:
            raise NotImplementedError('strict linear bound propagation not supported for module type')
    return new_build

factory.build = types.MethodType(overwrite_build(factory.build), factory)

__all__ = [
    'ScalarMult',
    'ScalarAdd',
    'Sum',
    'SqNorm',
    'factory',
    'BoundSin'
]
