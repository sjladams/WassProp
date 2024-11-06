from typing import Optional
import torch
from bound_propagation import BoundModelFactory, HyperRectangle, Parallel, VectorSub, IntervalBounds, LinearBounds, Mul, Reciprocal

from regions import HyperRectangularVoronoiPartition
from torch_modules import SqNorm
from optimize import minimize_with_adam

factory = BoundModelFactory()

def get_proj_matrix(vp: HyperRectangularVoronoiPartition):
    """
    Compute proj_{R_k}(c_i) for all signature locations c_i and regions R_k in the voronoi partition, and store
    in matrix with i-th row corresponding to c_i and k-th column corresponding to R_k.

    CONVENTION: indexing regions over columns and centers over rows

    :param vp: VornoiPartition
    """
    locs_ex = vp.locs.unsqueeze(-2)
    l_ex = vp.lower.unsqueeze(-3)
    u_ex = vp.upper.unsqueeze(-3)

    # Compute the projections for all locs and regions that do not overlap, set overlapping to nan
    mask_c_smaller_l = locs_ex <= l_ex
    mask_c_larger_u = locs_ex >= u_ex
    mask_c_in_region = torch.logical_and(~mask_c_larger_u, ~mask_c_smaller_l).all(-1)

    below_l = torch.where(mask_c_smaller_l, l_ex, torch.zeros_like(l_ex))
    above_u = torch.where(mask_c_larger_u, u_ex, torch.zeros_like(l_ex))
    overlapping = torch.where(mask_c_in_region.unsqueeze(-1).repeat(1,1,vp.locs.size(-1)),
                              torch.zeros_like(locs_ex).fill_(torch.nan),
                              torch.zeros_like(locs_ex))

    # Calculate the projection, summing both below and above cases
    proj_matrix = below_l + above_u + overlapping

    # Account for proj_{R_i}(c_i) = c_i
    proj_matrix.diagonal(dim1=-3, dim2=-2).copy_(vp.locs.swapaxes(-1, -2))

    # Handle non-overlapping regions:
    if vp.shell[:,0].isneginf().all() and vp.shell[:,1].isinf().all():
        proj_matrix[:, -1] = vp.locs  # To guarantee numerical stability, we set the projection on an empty set to zero
    else:
        closest_edge_shell = torch.where(
            (vp.locs - vp.shell[..., 0]).abs() < (vp.locs - vp.shell[..., 1]).abs(),
            vp.shell[..., 0],
            vp.shell[..., 1]
        )
        proj_matrix[:-1, -1] = closest_edge_shell[:-1]

    return torch.nan_to_num(proj_matrix)

def get_norm_of_proj_matrix(vp: HyperRectangularVoronoiPartition):
    """
    Compute ||proj_{R_k}(c_i) - c_i||_2 for all signature locations c_i and regions R_k in the voronoi
    partition, and store in matrix with i-th row corresponding to c_i and k-th column corresponding to R_k.

    :param vp: VoronoiPartition
    """
    proj_matrix = get_proj_matrix(vp)
    return torch.norm(proj_matrix - vp.locs.unsqueeze(-2), dim=-1, p=2)


class SqNormFxSubFz(torch.nn.Sequential):
    def __init__(self, f):
        super().__init__(
            Parallel(f, f, split_size=f.num_dims),
            VectorSub(),
            SqNorm(f.num_dims)
        )


@torch.no_grad()
def local_ibp_sq_norm_fx_fc(f: torch.nn.Sequential, vp: HyperRectangularVoronoiPartition) -> IntervalBounds:
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

    return sq_norm_fx_z.ibp(HyperRectangle(l_locs, u_locs))


def replace_inf_with(tensor: torch.Tensor, value: float=1e6):
    return tensor.masked_fill(torch.isinf(tensor), value)

def replace_neginf_with(tensor, value=-1e6):
    return tensor.masked_fill(torch.isneginf(tensor), value)


@torch.no_grad()
def global_ibp_sq_norm_fx_fc(f: torch.nn.Sequential, vp: HyperRectangularVoronoiPartition) -> IntervalBounds:
    """
    find vector b such that ||f(x) - f(c_i)||^2 leq b_i for all x  and c_i the loc of region R_i

    :param f: dynamics
    :param vp: VoronoiPartition
    """
    sq_norm_fx_z = factory.build(SqNormFxSubFz(f))

    l = torch.ones(vp.num_locs, f.num_dims).fill_(-torch.inf)
    u = torch.ones(vp.num_locs, f.num_dims).fill_(torch.inf)

    l = replace_inf_with(replace_neginf_with(l))  # \TODO check why this is needed:
    u = replace_inf_with(replace_neginf_with(u))

    l_locs = torch.cat((l, vp.locs), dim=-1)
    u_locs = torch.cat((u, vp.locs), dim=-1)

    ibp_bound = sq_norm_fx_z.ibp(HyperRectangle(l_locs, u_locs))
    return ibp_bound


def global_lbp_sq_norm_fx_fc(
        f: torch.nn.Sequential,
        vp: HyperRectangularVoronoiPartition,
        use_lbp: bool = True,
        beta: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    find vector a such that ||f(x) - f(c_i)||^2 leq a_i||x-c_i|| for all x and c_i the loc of region R_i

    :param f: dynamics
    :param vp: VoronoiPartition
    """

    if beta is None:
        beta = torch.zeros(vp.num_locs)

    if use_lbp:
        # the procedure below only works for "independent" dimension, to be defined properly

        # negative quadrant:
        input_bound_neg = HyperRectangle(torch.ones(vp.num_locs, vp.num_dims).fill_(-10.), vp.locs) # \todo to be set to -inf
        lb_neg = factory.build(f).crown(input_bound_neg)
        alpha_neg = torch.max(
            torch.svd(lb_neg.lower[0]).S.max(-1).values,
            torch.svd(lb_neg.upper[0]).S.max(-1).values
        ).pow(2)

        # positive quadrant:
        input_bound_pos = HyperRectangle(vp.locs, torch.ones(vp.num_locs, vp.num_dims).fill_(10.)) # \todo to be set to inf
        lb_pos = factory.build(f).crown(input_bound_pos)
        alpha_pos = torch.max(
            torch.svd(lb_pos.lower[0]).S.max(-1).values,
            torch.svd(lb_pos.upper[0]).S.max(-1).values
        ).pow(2)

        alpha = torch.max(alpha_neg, alpha_pos).clamp(min=0., max=f.global_lipschitz**2)
    else:
        # below we use a non-formal optimization based method. Using the bound-propagation package result in very-
        # conservative results

        def compute_local_lipschitz(x):
            local_lipschitz = ((f(x) - f(vp.locs)).pow(2).sum(-1) - beta) / (x - vp.locs).pow(2).sum(-1)
            local_lipschitz = torch.nan_to_num(local_lipschitz, nan=f.global_lipschitz ** 2)
            return local_lipschitz

        def objective(x):
            return - compute_local_lipschitz(x).sum() # or take mean?

        x_opt, losses = minimize_with_adam(
            objective,
            param=(vp.locs.clone().detach() + torch.randn_like(vp.locs)).requires_grad_(True),
            lr=0.01,
            num_iterations=5000,
            tolerance=1e-8,
            print_progress=True
        )

        alpha = compute_local_lipschitz(x_opt).detach().clamp(min=0., max=f.global_lipschitz**2)

    return alpha