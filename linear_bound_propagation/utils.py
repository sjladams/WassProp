import bound_propagation as bp
import torch


class NotSupportedError(Exception):
    def __init__(self, detail=""):
        base_message = ("Intersection is not a vertice of the bounds. So this procedure can not provide a strictly "
                        "linear bound at the intersection. Consider splitting it in compositional blocks")
        if detail:
            full_message = f"{base_message} Details: {detail}"
        else:
            full_message = base_message
        super().__init__(full_message)

def is_vertice(interval_bounds: bp.IntervalBounds, point):
    return torch.logical_or(
        torch.isclose(interval_bounds.lower, point, atol=1e-5),
        torch.isclose(point, interval_bounds.upper, atol=1e-5)
    ).all()

def _linear_map_intersect_at_point(A, b, point, y):
    """
    Checks if affine bound is linear around locs, i.e., check if A*point + b = y
    """
    bias = torch.einsum('...ij,...j->...i', A, point) + b - y
    return (bias.abs() <= 1e-5).all()

def linear_bounds_intersect_at_point(linear_bounds, point, y):
    msg_tmpl = ("{} bound does not intersect at point {}.\n Note that this exception is a last. "
                "Carefully check the method of BoundModules related to the strict procedure. Start with the "
                "strict_alpha_beta of the Activation functions that you used")
    assert _linear_map_intersect_at_point(*linear_bounds.lower, point, y), msg_tmpl.format("Lower", point)
    assert _linear_map_intersect_at_point(*linear_bounds.upper, point, y), msg_tmpl.format("Upper", point)