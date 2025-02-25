import torch
import bound_propagation as bp
from .linear import Linear

__all__ = ['ScalarMult', 'ScalarAdd', 'Sum', 'SqNorm']

class ScalarMult(Linear): #\todo rename MultScalar
    def __init__(self, in_features: int, scalar: float):
        super(ScalarMult, self).__init__(torch.eye(in_features) * scalar)

class ScalarAdd(Linear): # \todo rename AddScalar
    def __init__(self, in_features: int, scalar: float):
        super(ScalarAdd, self).__init__(torch.eye(in_features), torch.as_tensor(scalar))

class Sum(Linear):
    def __init__(self, in_features: int):
        super(Sum, self).__init__(torch.ones(1, in_features))

class SqNorm(torch.nn.Sequential):
    def __init__(self, num_dims):
        super().__init__(bp.Pow(2), Sum(num_dims))