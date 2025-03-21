import bound_propagation as bp
import torch
import types

from .arithmetic import MultScalar, AddScalar, Sum
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

__all__ = [
    'MultScalar',
    'AddScalar',
    'Sum',
    'factory',
    'BoundSin'
]
