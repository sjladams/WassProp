from typing import Optional
import torch
import bound_propagation as bp
from .utils import is_vertice, NotSupportedError

__all__ = ['BoundLinear', 'Linear', 'Identity']

class Linear(torch.nn.Linear):
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(weight.size(-1), weight.size(-2), bias=bias is not None)
        with torch.no_grad():
            self.weight.copy_(weight)
            if bias is not None:
                self.bias.copy_(bias)

class Identity(Linear):
    def __init__(self, in_features: int):
        super().__init__(torch.eye(in_features))

class BoundLinear(bp.BoundLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @bp.activation.assert_bound_order
    def strict_ibp_forward(self, bounds, intersection, save_relaxation=False, save_input_bounds=False):
        bounds = self.ibp_forward(bounds, save_relaxation, save_input_bounds)
        intersection = self(intersection)
        if not is_vertice(bounds, intersection):
            raise NotSupportedError()

        return bounds, intersection
