import bound_propagation as bp
import torch


def check_if_affine_bound_intersect_at_point(A, b, point, y):
    """
    Checks if affine bound is linear around locs, i.e., check if A*locs + b = y_locs
    """
    bias = torch.einsum('...ij,...j->...i', A, point) + b - y
    return (bias.abs() <= 1e-5).all()


class BoundSequential(bp.BoundSequential):
    def __init__(self, *args, **kwargs):
        super(BoundSequential, self).__init__(*args, **kwargs)

    def crown(self, *args, **kwargs):
        raise NotImplementedError

    def crown_ibp(self, *args, **kwargs):
        raise NotImplementedError

    def ibp(self, region):
        raise NotImplementedError

    def crown_ibp_point(self, region, point, bound_lower=True, bound_upper=True):
        linear_bounds = self.crown_with_relaxation(self.ibp_relax, region, bound_lower, bound_upper, alpha=False)

        y = self(point)

        msg_tmpl = "{} bound in {}-{} \n QUADRANT IS NOT LINEAR. Check BoundModules for dynamics or use Gradient Descent"
        assert check_if_affine_bound_intersect_at_point(*linear_bounds.lower, point, y), \
            msg_tmpl.format("Lower", region.lower, region.upper)
        assert check_if_affine_bound_intersect_at_point(*linear_bounds.upper, point, y), \
            msg_tmpl.format("Upper", region.lower, region.upper)
        return linear_bounds



