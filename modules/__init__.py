import bound_propagation as bp
import torch

from .arithmetic import ScalarMult, ScalarAdd, Sum, SqNorm
from .linear import BoundLinear, Linear
from .sigmoid import BoundSigmoid
from .saturation import BoundClamp
from .sin import BoundSin

linear_factory = bp.BoundModelFactory()
linear_factory.register(torch.nn.Sigmoid, BoundSigmoid)
linear_factory.register(Linear, BoundLinear)
linear_factory.register(bp.Clamp, BoundClamp)
linear_factory.register(bp.Sin, BoundSin)

__all__ = [
    'ScalarMult',
    'ScalarAdd',
    'Sum',
    'SqNorm',
    'linear_factory',
    'BoundSin'
]
