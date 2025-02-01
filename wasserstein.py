import torch
from typing import Callable, Union
import warnings
import dynamics
import discretize_distributions as ds
from regions import HyperRectangularVoronoiPartition
from bound import local_ibp_sq_norm_fx_fc, global_ibp_sq_norm_fx_fc, global_lbp_sq_norm_fx_fc
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
            signature: Union[ds.DiscretizedMultivariateNormal, ds.CategoricalFloat],
            f: dynamics.Dynamics,
            w2_q__disc_q: float,
            w2_p__q: float,
            **kwargs):

        if w2_p__q == 0:
            if isinstance(signature, ds.DiscretizedMultivariateNormal):
                fn_sq_w2_f_q__f_disc_q = get_fn_sq_w2_f_q__f_disc_q(signature, f)
                return fn_sq_w2_f_q__f_disc_q().sqrt()
            elif isinstance(signature, ds.CategoricalFloat):
                warnings.warn("Currently considering supports are equal so that quantization error is zero",
                              UserWarning)
                return 0.0
            else:
                raise NotImplementedError("Only implemented for DiscretizedMultivariateNormal and CategoricalFloat.")
        else:
            return func(signature, f, w2_q__disc_q, w2_p__q, **kwargs)
    return wrapper


def get_fn_sq_w2_f_q__f_disc_q(
        signature: Union[ds.DiscretizedMultivariateNormal, ds.CategoricalFloat],
        f: dynamics.Dynamics) -> Callable:

    if isinstance(signature, ds.DiscretizedMultivariateNormal):
        if not check_mat_diag(signature.dist.covariance_matrix):
            # To generalize beyond this, we have to implement non hyper-rectangular voronoi partitions, which as long as q
            # is a multivariate normal, AND not compression is applied, this is possible, because then the disc_q will be a
            # grid in the transformed space induced by the whitening transformation of the covariance matrix.
            raise NotImplementedError(
                "Only implemented for q being of the class MultivariateNormal with diagonal covariance matrix")

        voronoi_partition = HyperRectangularVoronoiPartition(signature.locs)

        alpha = global_lbp_sq_norm_fx_fc(f, signature.locs)
        beta = global_ibp_sq_norm_fx_fc(f, signature.locs).upper.squeeze(-1)

        sq_norm_2nd_moment = compute_sq_norm_2nd_moment(signature, voronoi_partition, signature.locs)

        def fn_sq_w2_f_q__f_disc_q():
            w2_alpha_or_beta = torch.min(sq_norm_2nd_moment * alpha, beta)
            return torch.einsum('...i,...i->...', w2_alpha_or_beta, signature.probs)

    elif isinstance(signature, ds.CategoricalFloat):
        raise NotImplementedError("Only implemented for q being of the class MultivariateNormal")

    else:
        raise NotImplementedError("Only implemented for q being of the class MultivariateNormal")

    return fn_sq_w2_f_q__f_disc_q


# ----- W_2(f#p, f#disc#q) for Lagrangian Duality approach -----
def get_fn_sq_w2_f_p__f_disc_q_lagrangian_duality(
        signature: Union[ds.DiscretizedMultivariateNormal, ds.CategoricalFloat],
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float)-> Callable:

    def fn_sq_w2_f_p__f_disc_q_lagrangian_duality(**kwargs): # \todo respect batches
        locs = signature.locs

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
        signature: Union[ds.DiscretizedMultivariateNormal, ds.CategoricalFloat],
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float,
        **kwargs):

    fn_sq_w2_f_p__f_disc_q = get_fn_sq_w2_f_p__f_disc_q_lagrangian_duality(signature, f, w2_q__disc_q, w2_p__q)
    w2 = fn_sq_w2_f_p__f_disc_q().sqrt()

    return w2