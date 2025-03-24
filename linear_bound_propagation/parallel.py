import bound_propagation as bp
import torch

__all__ = ['BoundParallel']

class BoundParallel(bp.BoundParallel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def strict_ibp_forward(self, bounds, intersection, save_relaxation=False, save_input_bounds=False):
        if self.module.split_size is None:
            split_bounds = [bounds for _ in range(len(self.subnetworks))]
            split_intersections = [intersection for _ in range(len(self.subnetworks))]
        else:
            split_bounds = self.split(bounds, self.split_sizes(bounds.lower.size(-1)))
            split_intersections = self.split(intersection, self.split_sizes(intersection.size(-1)))

        residuals = [network.strict_ibp_forward(bound, intersection, save_relaxation=save_relaxation, save_input_bounds=save_input_bounds)
                    for network, bound, intersection in zip(self.subnetworks, split_bounds, split_intersections)]

        lower = torch.cat([residual[0].lower for residual in residuals], dim=-1)
        upper = torch.cat([residual[0].upper for residual in residuals], dim=-1)
        bounds = bp.IntervalBounds(bounds.region, lower, upper)

        intersection = torch.cat([residual[1] for residual in residuals], dim=-1)

        return bounds, intersection
