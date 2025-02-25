from typing import Optional
import torch
import bound_propagation as bp

from dynamics import LinearDiagonalDynamics, LinearDiagonalBoundedDynamics, AdditiveGaussianDynamics
from regions import HyperRectangularVoronoiPartition
from linear_bound_propagation import SqNorm, factory as linear_factory
from optimize import minimize_with_adam
from tensors import check_mat_diag

factory = bp.BoundModelFactory()


class SqNormFxSubFz(torch.nn.Sequential):
    def __init__(self, f):
        super().__init__(
            bp.Parallel(f, f, split_size=f.num_dims),
            bp.VectorSub(),
            SqNorm(f.num_state_dims if hasattr(f, 'num_state_dims') else f.num_dims)
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


def _global_lbp_sq_norm_fx_fc_quadrant(
        f: torch.nn.Sequential,
        locs: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        independent_dims: bool = False) -> torch.Tensor:

    input_bound = bp.HyperRectangle(lower, upper)
    lb = linear_factory.build(f).crown_ibp_point(input_bound, locs)

    # From linear bounds to bounds on the norms:
    alpha = torch.max(
        torch.svd(lb.lower[0]).S.max(-1).values,
        torch.svd(lb.upper[0]).S.max(-1).values
    ).pow(2)

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
    else:
        # below we use a non-formal optimization based method. Using the bound-propagation package result in very-
        # conservative results

        def compute_local_lipschitz(x):
            local_lipschitz = ((f(x) - f(locs)).pow(2).sum(-1) - beta) / (x - locs).pow(2).sum(-1)
            local_lipschitz = torch.nan_to_num(local_lipschitz, nan=f.global_lipschitz ** 2)
            return local_lipschitz

        def objective(x):
            return - compute_local_lipschitz(x).sum() # or take mean?

        x_opt, losses = minimize_with_adam(
            objective,
            param=(locs.clone().detach() + torch.randn_like(locs)).requires_grad_(True),
            lr=0.01,
            num_iterations=5000,
            tolerance=1e-8,
            print_progress=True
        )

        alpha = compute_local_lipschitz(x_opt).detach().clamp(min=0., max=f.global_lipschitz**2)

    return alpha