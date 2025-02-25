import torch
import bound_propagation as bp

__all__ = ['ScalarMult', 'ScalarAdd', 'Sum', 'SqNorm']

class ScalarMult(torch.nn.Linear): #\todo rename MultScalar
    def __init__(self, in_features: int, scalar: float):
        super(ScalarMult, self).__init__(in_features, in_features, bias=False)
        with torch.no_grad():
            self.weight.copy_(torch.eye(in_features) * scalar)

class ScalarAdd(torch.nn.Linear): # \todo rename AddScalar
    def __init__(self, in_features: int, scalar: float):
        super(ScalarAdd, self).__init__(in_features, in_features)
        with torch.no_grad():
            self.weight.copy_(torch.eye(in_features))
            self.bias.fill_(scalar)

class Sum(torch.nn.Linear):
    def __init__(self, in_features: int):
        super(Sum, self).__init__(in_features, 1, bias=False)
        with torch.no_grad():
            self.weight.fill_(1.0)

class SqNorm(torch.nn.Sequential):
    def __init__(self, num_dims):
        super().__init__(bp.Pow(2), Sum(num_dims))