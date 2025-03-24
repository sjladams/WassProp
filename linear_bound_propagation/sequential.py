import bound_propagation as bp
import torch
from .utils import linear_bounds_intersect_at_point

class BoundSequential(bp.BoundSequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def strict_crown_ibp(self, region, intersection, bound_lower=True, bound_upper=True):
        out_size = self.propagate_size(region.size(-1))
        self.strict_ibp_relax(region, intersection)

        linear_bounds = self.initial_linear_bounds(region, out_size, lower=bound_lower, upper=bound_upper)
        linear_bounds = self.crown_backward(linear_bounds, False)

        self.clear_relaxation()

        linear_bounds_intersect_at_point(linear_bounds, intersection, self.module(intersection))

        return linear_bounds

    @torch.no_grad()
    def strict_ibp_relax(self, region, intersection):
        bounds = region.bounding_hyperrect()
        self.strict_ibp_forward(bounds, intersection, save_relaxation=True)

    def strict_ibp_forward(self, bounds, intersection, save_relaxation=False, save_input_bounds=False):
        for module in self.bound_sequential:
            bounds, intersection = module.strict_ibp_forward(bounds, intersection, save_relaxation=save_relaxation, save_input_bounds=save_input_bounds)
        return bounds, intersection
