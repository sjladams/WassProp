from typing import Optional
import torch
import bound_propagation as bp

from dynamics import LinearDiagonalDynamics, LinearDiagonalBoundedDynamics, AdditiveGaussianDynamics
from regions import HyperRectangularVoronoiPartition
from modules import SqNorm, linear_factory
from tensors import check_mat_diag
import warnings

factory = bp.BoundModelFactory()

class SqNormFxSubFz(torch.nn.Sequential):
    def __init__(self, f):
        super().__init__(
            bp.Parallel(f, f, split_size=f.num_dims),
            bp.VectorSub(),
            SqNorm(f.num_dims)
        )


@torch.no_grad()
def local_ibp_sq_norm_fx_fc(f: torch.nn.Sequential, vp: HyperRectangularVoronoiPartition) -> bp.IntervalBounds:
    """
    find matrix B such that ||f(x) - f(c_i)||^2 leq B^{(ik)} for all x in region [l_k, u_k] and c_i the loc
     of region R_i

    :param f: dynamics
    :param vp: VoronoiPartition

    """
    sq_norm_fx_z = factory.build(SqNormFxSubFz(f))

    l = replace_inf_with(replace_neginf_with(vp.lower))   # \TODO check why this is needed:
    u = replace_inf_with(replace_neginf_with(vp.upper))

    l_locs = torch.cat((l.unsqueeze(-3).repeat(vp.num_locs, 1, 1), vp.locs.unsqueeze(-2).repeat(1, vp.num_locs, 1)), dim=-1)
    u_locs = torch.cat((u.unsqueeze(-3).repeat(vp.num_locs, 1, 1), vp.locs.unsqueeze(-2).repeat(1, vp.num_locs, 1)), dim=-1)

    return sq_norm_fx_z.ibp(bp.HyperRectangle(l_locs, u_locs))


def replace_inf_with(tensor: torch.Tensor, value: float=1e6):
    return tensor.masked_fill(torch.isinf(tensor), value)

def replace_neginf_with(tensor, value=-1e6):
    return tensor.masked_fill(torch.isneginf(tensor), value)


def global_ibp_sq_norm_fx_fc(f: torch.nn.Sequential, locs: torch.Tensor) -> bp.IntervalBounds:
    """
    find vector b such that ||f(x) - f(c_i)||^2 leq b_i for all x  and c_i the loc of region R_i

    :param f: dynamics
    :param locs: c_i's
    """
    num_locs = locs.shape[-2]

    sq_norm_fx_z = factory.build(SqNormFxSubFz(f))

    l = torch.ones(num_locs, f.num_dims).fill_(-torch.inf)
    u = torch.ones(num_locs, f.num_dims).fill_(torch.inf)

    l = replace_inf_with(replace_neginf_with(l))  # \TODO check why this is needed:
    u = replace_inf_with(replace_neginf_with(u))

    l_locs = torch.cat((l, locs), dim=-1)
    u_locs = torch.cat((u, locs), dim=-1)

    ibp_bound = sq_norm_fx_z.ibp(bp.HyperRectangle(l_locs, u_locs))
    return ibp_bound


def check_if_affine_bound_is_linear_at_locs(A, b, locs, y_locs):
    """
    Checks if affine bound is linear around locs, i.e., check if A*locs + b = y_locs
    """
    bias = torch.einsum('nij,nj->ni', A, locs) + b - y_locs
    return (bias.abs() <= 1e-5).all()


def _global_lbp_sq_norm_fx_fc_quadrant(
        f: torch.nn.Sequential,
        locs: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        independent_dims: bool = False) -> torch.Tensor:

    input_bound = bp.HyperRectangle(lower, upper)
    lb = linear_factory.build(f).crown_ibp(input_bound)

    #assert independent_dims or check_mat_diag(lb.lower[0]) and check_mat_diag(lb.upper[0]), \
    #    "Currently global_lbp_sq_norm_fc only works for independent dimensions"
    warnings.warn("CHECK TESTS",
                  UserWarning) #TODO: Check

    # From linear bounds to bounds on the norms:
    alpha = torch.max(
        torch.svd(lb.lower[0]).S.max(-1).values,
        torch.svd(lb.upper[0]).S.max(-1).values
    ).pow(2)

    y_locs = f(locs)
    msg_tmpl = "{} bound in {}-{} \n QUADRANT IS NOT LINEAR. Check BoundModule for dynamics or use Gradient Descent"
    if not check_if_affine_bound_is_linear_at_locs(lb.lower[0], lb.lower[1], locs, y_locs):
        check_if_affine_bound_is_linear_at_locs(lb.lower[0], lb.lower[1], locs, y_locs)
        print(msg_tmpl.format("Lower", lower, upper))
    assert check_if_affine_bound_is_linear_at_locs(lb.upper[0], lb.upper[1], locs, y_locs), \
        msg_tmpl.format("Upper", lower, upper)

    return alpha


def global_lbp_sq_norm_fx_fc(
        f: torch.nn.Sequential,
        locs: torch.Tensor,
        use_lbp: bool = True,
        beta: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    find vector a such that ||f(x) - f(c_i)||^2 leq a_i||x-c_i||^2 for all x and c_i the loc of region R_i

    :param f: dynamics
    :param locs: batch of c's with shape (num_locs, num_dims)
    """

    num_locs = locs.shape[-2]
    num_dims = locs.shape[-1]

    if beta is None:
        beta = torch.zeros(num_locs)

    if use_lbp:
        # quadrants of shape (nr_quadrants, 2, num_locs, num_dims)
        if isinstance(f, (LinearDiagonalDynamics, LinearDiagonalBoundedDynamics, )):
            # If the dynamics is separable, then we only have to the positive and negative quadrants
            quadrants = torch.stack((
                torch.stack((torch.ones(num_locs, num_dims).fill_(-torch.inf), locs)),
                torch.stack((locs, torch.ones(num_locs, num_dims).fill_(torch.inf)))
            ))
            independent_dims = True
        else:
            # else, we have to consider all quadrants:
            # Generate all combinations of signs (+1 and -1) for each dimension
            signs = torch.cartesian_prod(*[torch.tensor([-1, 1]) for _ in range(num_dims)])

            # Compute lower and upper bounds for each quadrant
            lower = torch.where(signs==-1, torch.full_like(signs, -torch.inf, dtype=locs.dtype), torch.zeros_like(signs, dtype=locs.dtype))
            upper = torch.where(signs==1, torch.full_like(signs, torch.inf, dtype=locs.dtype), torch.zeros_like(signs, dtype=locs.dtype))

            # Stack lower and upper bounds into shape (nr_quadrants, 2, n)
            quadrants = torch.stack([lower, upper], dim=1)
            quadrants = quadrants.unsqueeze(-2).repeat(1, 1, num_locs, 1) + locs

            independent_dims = False

        alphas = torch.zeros(len(quadrants), num_locs).fill_(torch.nan)
        for idx, quadrant in enumerate(quadrants):
            alphas[idx] = _global_lbp_sq_norm_fx_fc_quadrant(f, locs, quadrant[0], quadrant[1], independent_dims )

        alpha = alphas.max(dim=0).values.clamp(min=0., max=f.global_lipschitz**2)

    return alpha