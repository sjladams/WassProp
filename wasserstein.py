import torch
from typing import Callable, Union

import dynamics
import discretize_distributions as ds
from regions import HyperRectangularVoronoiPartition

from bound import local_ibp_sq_norm_fx_fc, global_ibp_sq_norm_fx_fc, global_lbp_sq_norm_fx_fc

from optimize import minimize_with_adam
from tensors import check_mat_diag

def compute_sq_norm_2nd_moment(signature: ds.DiscretizedMultivariateNormal, voronoi_partition: HyperRectangularVoronoiPartition, locs: torch.Tensor):
    # \todo include in discretize_distributions packages
    # \todo locs are the representative points of the partition, which in case of a shift is not the voronoi_partition any more: Create new partition class that combines both locs and lower and upper

    ## Compute integral terms:
    trunc_mean, trunc_var = ds.utils.calculate_mean_and_var_trunc_normal(
        loc=signature.dist.loc.unsqueeze(0),
        scale=signature.dist.covariance_matrix.diagonal(dim1=-1, dim2=-2).sqrt().unsqueeze(0),
        l=voronoi_partition.lower, u=voronoi_partition.upper)

    return (trunc_var + (trunc_mean - locs).pow(2)).sum(-1)


def compute_w2_wrapper(func):
    def wrapper(
            signature: ds.DiscretizedMultivariateNormal,
            f: dynamics.Dynamics,
            w2_q__disc_q: float,
            w2_p__q: float,
            **kwargs):

        if w2_p__q == 0 and isinstance(signature, ds.DiscretizedMultivariateNormal):
            fn_sq_w2_f_q__f_disc_q = get_fn_sq_w2_f_q__f_disc_q(signature, f)

            return fn_sq_w2_f_q__f_disc_q().sqrt()
        else:
            return func(signature, f, w2_q__disc_q, w2_p__q, **kwargs)
    return wrapper


def get_fn_sq_w2_f_q__f_disc_q(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics) -> Callable:
    if not isinstance(signature.dist, ds.MultivariateNormal):
        raise NotImplementedError("Only implemented for q being of the class MultivariateNormal")

    if not check_mat_diag(signature.dist.covariance_matrix):
        # To generalize beyond this, we have to implement non hyper-rectangular voronoi partitions, which as long as q
        # is a multivariate normal, AND not compression is applied, this is possible, because then the disc_q will be a
        # grid in the transformed space induced by the whitening transformation of the covariance matrix.
        raise NotImplementedError("Only implemented for q being of the class MultivariateNormal with diagonal covariance matrix")

    voronoi_partition = HyperRectangularVoronoiPartition(signature.locs)

    alpha = global_lbp_sq_norm_fx_fc(f, signature.locs)
    beta = global_ibp_sq_norm_fx_fc(f, signature.locs).upper.squeeze(-1)

    sq_norm_2nd_moment = compute_sq_norm_2nd_moment(signature, voronoi_partition, signature.locs)

    def fn_sq_w2_f_q__f_disc_q():
        w2_alpha_or_beta = torch.min(sq_norm_2nd_moment * alpha, beta)
        return torch.einsum('...i,...i->...', w2_alpha_or_beta, signature.probs)

    return fn_sq_w2_f_q__f_disc_q


# ----- W_2(f#p, f#disc#q) for Lagrangian Duality approach -----
def get_fn_sq_w2_f_p__f_disc_q_lagrangian_duality(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float)-> Callable:

    def fn_sq_w2_f_p__f_disc_q_lagrangian_duality(locs_shift: Union[torch.Tensor, float] = 0., **kwargs): # \todo respect batches
        locs = signature.locs + locs_shift

        if (torch.as_tensor(locs_shift) != 0.).any():
            if not isinstance(f, dynamics.NonAdditiveGaussianNoiseDynamics):
                voronoi_partition = HyperRectangularVoronoiPartition(signature.locs)

                sq_norm_2nd_moment = compute_sq_norm_2nd_moment(signature, voronoi_partition, locs) # we use the same partition and probs, only move the locations
                w2_disc = torch.sum(sq_norm_2nd_moment * signature.probs).sqrt()
                w2_p__disc_q = w2_p__q + w2_disc
            else:
                raise NotImplementedError
        else:
            w2_p__disc_q = w2_p__q + w2_q__disc_q

        alpha = global_lbp_sq_norm_fx_fc(f, locs)
        beta = global_ibp_sq_norm_fx_fc(f, locs).upper.squeeze(-1)

        mask = torch.ones(alpha.size(0), alpha.size(0)).tril()[:, alpha.sort().indices]
        mask = torch.cat((mask, torch.zeros(1, locs.shape[-2])), dim=0)

        alpha_options = torch.einsum('ij, j->ij', mask, alpha)
        beta_options = torch.einsum('ij, j->ij', 1 - mask, beta)

        alpha_max = alpha_options.max(dim=-1).values
        result_options = alpha_max * w2_p__disc_q ** 2 + torch.einsum('j,ij->i', signature.probs, beta_options)

        return result_options.min(-1).values

    return fn_sq_w2_f_p__f_disc_q_lagrangian_duality


@compute_w2_wrapper
def compute_w2_f_p__f_disc_q_lagrangian_duality(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float,
        optimize_locs: bool = False,
        **kwargs):

    fn_sq_w2_f_p__f_disc_q = get_fn_sq_w2_f_p__f_disc_q_lagrangian_duality(signature, f, w2_q__disc_q, w2_p__q)

    if optimize_locs:
        locs_shift = torch.randn_like(signature.locs.detach()) * 5.0
        optimal_shift, losses = minimize_with_adam(
            param=locs_shift.requires_grad_(True),
            objective=fn_sq_w2_f_p__f_disc_q,
            **kwargs
        )
        w2 = fn_sq_w2_f_p__f_disc_q(optimal_shift).sqrt()
        print(f'w2 after GD (starting from signatures + gaussian noise): {w2:.4f}')
        w2_zero_shift = fn_sq_w2_f_p__f_disc_q().sqrt()
        if w2_zero_shift < w2:
            w2 = w2_zero_shift
            print('Optimal shift is zero!')
        else:
            print(f'w2 improvement due to loc optimization: {w2-w2_zero_shift:.8f}')
    else:
        w2 = fn_sq_w2_f_p__f_disc_q().sqrt()

    return w2
