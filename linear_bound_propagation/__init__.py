import bound_propagation as bp
import torch

from .arithmetic import MultScalar, AddScalar, Sum
from .linear import BoundLinear, Linear, Identity
from .activation import BoundSigmoid, BoundIdentity, BoxedIdentity, BoundBoxedIdentity
from .saturation import BoundClamp
from .sin import BoundSin
from .sequential import BoundSequential
from .parallel import BoundParallel
from .bivariate import BoundVectorAdd

factory = bp.BoundModelFactory()
factory.register(torch.nn.Sigmoid, BoundSigmoid)
factory.register(Linear, BoundLinear)
factory.register(bp.Clamp, BoundClamp)
factory.register(bp.Sin, BoundSin)
factory.register(torch.nn.Sequential, BoundSequential)
factory.register(torch.nn.Identity, BoundIdentity)
factory.register(bp.Parallel, BoundParallel)
factory.register(bp.VectorAdd, BoundVectorAdd)
factory.register(BoxedIdentity, BoundBoxedIdentity)

__all__ = [
    'MultScalar',
    'AddScalar',
    'Sum',
    'factory',
    'BoundSin',
    'BoxedIdentity'
]
