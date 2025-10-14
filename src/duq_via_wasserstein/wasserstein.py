import torch
from typing import Callable, Union

import discretize_distributions.distributions as dd_dists

from . import dynamics
from .bound import global_ibp_sq_norm_fx_fc, global_lbp_sq_norm_fx_fc


def get_fn_sq_w2_f_q__f_disc_q(
    q: dd_dists.MultivariateNormal,
    disc_q: dd_dists.CategoricalFloat,
    f: dynamics.StochasticDynamics
) -> Callable:
    if not isinstance(q, dd_dists.MultivariateNormal):
        raise NotImplementedError("Only implemented for q being of the class MultivariateNormal")

    if not check_mat_diag(q.covariance_matrix):
        # To generalize beyond this, we have to implement non hyper-rectangular voronoi partitions, which as long as q
        # is a multivariate normal, AND not compression is applied, this is possible, because then the disc_q will be a
        # grid in the transformed space induced by the whitening transformation of the covariance matrix.
        raise NotImplementedError("Only implemented for q being of the class MultivariateNormal with diagonal covariance matrix")

    alpha = global_lbp_sq_norm_fx_fc(f, disc_q.locs)
    beta = global_ibp_sq_norm_fx_fc(f, disc_q.locs)

    def fn_sq_w2_f_q__f_disc_q():
        w2_alpha_or_beta = torch.min(disc_q.sq_l2_norm * alpha, beta)
        return torch.einsum('...i,...i->...', w2_alpha_or_beta, disc_q.probs)

    return fn_sq_w2_f_q__f_disc_q


# ----- W_2(f#p, f#disc#q) for Lagrangian Duality approach -----
def get_fn_sq_w2_f_p__f_disc_q_lagrangian_duality(
    disc_q: dd_dists.CategoricalFloat,
    f: dynamics.StochasticDynamics,
    w2_p__disc_q: Union[float, torch.Tensor]
) -> Callable:

    def fn_sq_w2_f_p__f_disc_q_lagrangian_duality(**kwargs):
        alpha = global_lbp_sq_norm_fx_fc(f, disc_q.locs)
        beta = global_ibp_sq_norm_fx_fc(f, disc_q.locs)

        mask = torch.ones(alpha.size(0), alpha.size(0)).tril()[:, alpha.sort().indices]
        mask = torch.cat((mask, torch.zeros(1, disc_q.locs.shape[-2])), dim=0)

        alpha_options = torch.einsum('ij, j->ij', mask, alpha)
        beta_options = torch.einsum('ij, j->ij', 1 - mask, beta)

        alpha_max = alpha_options.max(dim=-1).values
        result_options = alpha_max * w2_p__disc_q ** 2 + torch.einsum('j,ij->i', disc_q.probs, beta_options)

        return result_options.min(-1).values

    return fn_sq_w2_f_p__f_disc_q_lagrangian_duality


def compute_w2_f_p__f_disc_q_lagrangian_duality(
    disc_q: dd_dists.CategoricalFloat,
    f: dynamics.StochasticDynamics,
    w2_p__disc_q: Union[float, torch.Tensor]
) -> torch.Tensor:
        
    ## TODO Implement Theorem 5.2 for the case where w2_p__q == 0 and isinstance(q, dd_dists.MixtureMultivariateNormal) 
    # using get_fn_sq_w2_f_q__f_disc_q, which requires computing, or pulling the sq_l2_norm during the discretizaiton operation

    fn_sq_w2_f_p__f_disc_q = get_fn_sq_w2_f_p__f_disc_q_lagrangian_duality(disc_q, f, w2_p__disc_q)
    w2 = fn_sq_w2_f_p__f_disc_q().sqrt()

    return w2



PRECISION = torch.finfo(torch.float32).eps

def check_mat_diag(mat: torch.Tensor) -> bool:
    """
    Check if all elements of a batch of square matrices are diagonal
    """
    if mat.shape[-1] != mat.shape[-2]:
        return False
    else:
        return ((mat - torch.diag_embed(mat.diagonal(dim1=-1, dim2=-2), dim1=-1, dim2=-2)).abs() < PRECISION).all()