from typing import Optional
import torch

__all__ = ['MultScalar', 'AddScalar', 'Sum']

class Linear(torch.nn.Linear):
    def __init__(
        self, 
        weight: torch.Tensor, 
        bias: Optional[torch.Tensor] = None
    ):
        super().__init__(weight.size(-1), weight.size(-2), bias=bias is not None)
        with torch.no_grad():
            self.weight.copy_(weight)
            if bias is not None:
                self.bias.copy_(bias)

class MultScalar(Linear):
    def __init__(self, in_features: int, scalar: float):
        super().__init__(torch.eye(in_features) * scalar)

class AddScalar(Linear):
    def __init__(self, in_features: int, scalar: float):
        super().__init__(torch.eye(in_features), torch.as_tensor(scalar))

class Sum(Linear):
    def __init__(self, in_features: int):
        super().__init__(torch.ones(1, in_features))
