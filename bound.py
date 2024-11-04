import torch
from bound_propagation import Pow, BoundModelFactory, HyperRectangle, Parallel, VectorSub

from regions import HyperRectangularVoronoiPartition

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
    overlapping = torch.where(mask_c_in_region.unsqueeze(-1).repeat(1,1,2),
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

    return proj_matrix

def get_norm_of_proj_matrix(vp: HyperRectangularVoronoiPartition):
    """
    Compute ||proj_{R_k}(c_i) - c_i||_2 for all signature locations c_i and regions R_k in the voronoi
    partition, and store in matrix with i-th row corresponding to c_i and k-th column corresponding to R_k.

    :param vp: VoronoiPartition
    """
    proj_matrix = get_proj_matrix(vp)
    return torch.norm(proj_matrix - vp.locs.unsqueeze(-2), dim=-1, p=2)


class SqVecNorm(torch.nn.Sequential):
    def __init__(self, in_features: int):
        linear = torch.nn.Linear(in_features, 1, bias=False)
        with torch.no_grad():
            linear.weight.fill_(1.0)
        super().__init__(Pow(2), linear)


def bound_sq_norm_fx_fc(f: torch.nn.Sequential, vp: HyperRectangularVoronoiPartition):
    """
    find matrix B such that ||f(x) - f(c_i)||^2 leq B^{(ik)} for all x in region [l_k, u_k] and c_i the loc
     of region R_i

    :param f: dynamics
    :param vp: VoronoiPartition
    """
    num_locs = vp.locs.size(-2)

    sq_norm_fx_z = torch.nn.Sequential(
        Parallel(f, torch.nn.Identity(), split_size=f.num_dims),
        VectorSub(),
        SqVecNorm(f.num_dims)
    )
    sq_norm_fx_z = factory.build(sq_norm_fx_z)

    flocs = f(vp.locs)

    l_flocs = torch.cat((vp.lower.unsqueeze(-3).repeat(num_locs, 1, 1), flocs.unsqueeze(-2).repeat(1, num_locs, 1)), dim=-1)
    u_flocs = torch.cat((vp.upper.unsqueeze(-3).repeat(num_locs, 1, 1), flocs.unsqueeze(-2).repeat(1, num_locs, 1)), dim=-1)

    # \TODO check why this is needed:
    l_flocs = replace_inf_with(replace_neginf_with(l_flocs))
    u_flocs = replace_inf_with(replace_neginf_with(u_flocs))

    input_bounds = HyperRectangle(l_flocs.view(-1, f.num_dims*2), u_flocs.view(-1, f.num_dims*2))

    return sq_norm_fx_z.ibp(input_bounds).upper.view(num_locs, num_locs)


def bound_sq_norm_fx_fc_OLD(f: torch.nn.Sequential, vp: HyperRectangularVoronoiPartition):
    """
    find matrix B such that ||f(x) - f(c_i)||^2 leq B^{(ik)} for all x in region [l_k, u_k] and c_i the loc
     of region R_i

    :param f: dynamics
    :param vp:
    """

    lp2_norm_difference_upper = torch.norm(f(vp.upper).unsqueeze(0) - f(vp.locs).unsqueeze(1), p=2, dim=-1)
    lp2_norm_difference_lower = torch.norm(f(vp.lower).unsqueeze(0) - f(vp.locs).unsqueeze(1), p=2, dim=-1)
    return torch.max(lp2_norm_difference_upper, lp2_norm_difference_lower)


def replace_inf_with(tensor: torch.Tensor, value: float=1e6):
    return tensor.masked_fill(torch.isinf(tensor), value)

def replace_neginf_with(tensor, value=-1e6):
    return tensor.masked_fill(torch.isneginf(tensor), value)

