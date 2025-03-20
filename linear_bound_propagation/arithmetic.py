import torch
from .linear import Linear

__all__ = ['ScalarMult', 'ScalarAdd', 'Sum']

class ScalarMult(Linear): #\todo rename MultScalar
    def __init__(self, in_features: int, scalar: float):
        super().__init__(torch.eye(in_features) * scalar)

class ScalarAdd(Linear): # \todo rename AddScalar
    def __init__(self, in_features: int, scalar: float):
        super().__init__(torch.eye(in_features), torch.as_tensor(scalar))

class Sum(Linear):
    def __init__(self, in_features: int):
        super().__init__(torch.ones(1, in_features))
