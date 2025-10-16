from typing import Optional
import torch
import bound_propagation as bp
from .activation import assert_bound_order

__all__ = ['BoundLinear']


class BoundLinear(bp.BoundLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @assert_bound_order
    def strict_ibp_forward(self, bounds, intersection, save_relaxation=False, save_input_bounds=False):
        bounds = self.ibp_forward(bounds, save_relaxation, save_input_bounds)
        intersection = self.module(intersection)

        return bounds, intersection
