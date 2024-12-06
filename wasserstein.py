import torch
from typing import Callable, Union

import dynamics
import discretize_distributions as ds
from regions import HyperRectangularVoronoiPartition

from bound import local_ibp_sq_norm_fx_fc, get_norm_of_proj_matrix, global_ibp_sq_norm_fx_fc, global_lbp_sq_norm_fx_fc

from optimize import minimize_with_adam


def compute_w2_wrapper(func):
    def wrapper(
            signature: ds.DiscretizedMultivariateNormal,
            f: dynamics.Dynamics,
            w2_q__disc_q: float,
            w2_p__q: float,
            **kwargs):

        if w2_p__q == 0:
            fn_sq_w2_f_q__f_disc_q = get_fn_sq_w2_f_q__f_disc_q(signature, f)

            return fn_sq_w2_f_q__f_disc_q().sqrt()
        else:
            return func(signature, f, w2_q__disc_q, w2_p__q, **kwargs)
    return wrapper


# ----- W_2(f#p, f#disc#p) -----
def get_fn_sq_w2_f_p__f_disc_p(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float) -> Callable:

    voronoi_partition = HyperRectangularVoronoiPartition(signature.locs)

    sq_norm_fx_fc = local_ibp_sq_norm_fx_fc(f, voronoi_partition).upper.squeeze(-1).diag()

    sq_norm_proj_matrix = get_norm_of_proj_matrix(voronoi_partition).pow(2)

    w2_p__disc_q = w2_q__disc_q + w2_p__q

    def fn_sq_w2_f_p__f_disc_p(lambd: torch.Tensor):
        inner_sup = torch.max(sq_norm_fx_fc - lambd * sq_norm_proj_matrix, dim=-1).values
        return lambd * w2_p__disc_q ** 2 + torch.einsum('m,m->', signature.probs, inner_sup)

    return fn_sq_w2_f_p__f_disc_p


def get_fn_sq_w2_f_q__f_disc_q(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics) -> Callable:
    voronoi_partition = HyperRectangularVoronoiPartition(signature.locs)

    alphas = global_lbp_sq_norm_fx_fc(f, signature.locs)
    locs = signature.locs
    means = signature.dist.mean
    sigmas = signature.dist._sqrt_diag_covariance_matrix
    lower = voronoi_partition.lower
    upper = voronoi_partition.upper

    upper[torch.isposinf(upper)] = 1e4  # large value
    lower[torch.isneginf(lower)] = -1e4

    scaled_lower = (lower - means) / (2 ** 0.5 * sigmas)
    scaled_upper = (upper - means) / (2 ** 0.5 * sigmas)

    erf_lower = torch.special.erf(scaled_lower)
    erf_upper = torch.special.erf(scaled_upper)

    common_factor = 0.5 * ((locs - means) ** 2 + sigmas ** 2)

    def fn_sq_w2_f_p__f_disc_p():
        factor_exp_upper = (1 / (2 * torch.pi) ** 0.5) * sigmas * (upper - 2 * locs + means) * torch.exp(
            -scaled_upper ** 2)
        factor_exp_lower = (1 / (2 * torch.pi) ** 0.5) * sigmas * (lower - 2 * locs + means) * torch.exp(
            -scaled_lower ** 2)

        #Using Mathematica
        integral_per_dim = common_factor * (erf_upper - erf_lower) - factor_exp_upper + factor_exp_lower
        integral = torch.prod(integral_per_dim, dim=1)

        return torch.dot(alphas, integral)

    return fn_sq_w2_f_p__f_disc_p


@compute_w2_wrapper
def compute_w2_f_p__f_disc_p(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float,
        **kwargs):

    fn_sq_w2_f_p__f_disc_p = get_fn_sq_w2_f_p__f_disc_p(signature, f, w2_q__disc_q, w2_p__q)

    lambd = torch.tensor(0.1, requires_grad=True)
    optimized_lambda, losses = minimize_with_adam(
        param=lambd,
        objective=fn_sq_w2_f_p__f_disc_p,
        lower_constraint=0.,
        **kwargs)

    return fn_sq_w2_f_p__f_disc_p(optimized_lambda).sqrt()


# ----- W_2(f#p, f#disc#q) for Independent Coupling approach -----
def get_fn_sq_w2_f_p__f_disc_q_independent_coupling(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float) -> Callable:

    w2_p__disc_q = w2_q__disc_q + w2_p__q

    alpha = global_lbp_sq_norm_fx_fc(f, signature.locs)

    def fn_sq_w2_f_p__f_disc_q_independent_coupling(lambd: torch.Tensor): # \todo: respect batches
        v = lambd * signature.locs - ((signature.probs * alpha).unsqueeze(1) * signature.locs).sum(dim=0, keepdim=True)
        coeff_v = 1 / (lambd - torch.dot(signature.probs, alpha))

        c__transpose__c = torch.sum(signature.locs ** 2, dim=1)
        sum_pi_alpha_c__transpose__c = torch.sum(signature.probs * alpha * c__transpose__c)

        quadrat_sol = coeff_v * (v ** 2).sum(dim=1) - lambd * (signature.locs ** 2).sum(dim=1) + sum_pi_alpha_c__transpose__c

        return lambd * w2_p__disc_q ** 2 + torch.dot(signature.probs, quadrat_sol)

    return fn_sq_w2_f_p__f_disc_q_independent_coupling


@compute_w2_wrapper
def compute_w2_f_p__f_disc_q_independent_coupling(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float,
        **kwargs):
    # \todo revise independend coupling: use package on stable truncated gaussians, and get rid of doulbe alpha computation

    alpha = global_lbp_sq_norm_fx_fc(f, signature.locs)
    avg_alpha = torch.dot(alpha, signature.probs).detach()

    fn_sq_w2_f_p__f_disc_q = get_fn_sq_w2_f_p__f_disc_q_independent_coupling(
        signature, f, w2_q__disc_q, w2_p__q)

    optimized_lambda, losses = minimize_with_adam(
        param=avg_alpha + 10.,
        objective=fn_sq_w2_f_p__f_disc_q,
        lower_constraint=avg_alpha,
        **kwargs
    )

    return fn_sq_w2_f_p__f_disc_q(optimized_lambda).sqrt()


# ----- W_2(f#p, f#disc#q) for Lagrangian Duality approach -----
def get_fn_sq_w2_f_p__f_disc_q_lagrangian_duality(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float)-> Callable:

    w2_p__disc_q = w2_q__disc_q + w2_p__q

    def fn_sq_w2_f_p__f_disc_q_lagrangian_duality(locs_shift: Union[torch.Tensor, float] = 0.): # \todo respect batches
        locs = signature.locs + locs_shift
        w2_shift_locs = (locs - signature.locs).norm(p=2, dim=-1).pow(2).sum(-1)

        alpha = global_lbp_sq_norm_fx_fc(f, locs)
        beta = global_ibp_sq_norm_fx_fc(f, locs).upper.squeeze(-1)

        mask = torch.ones(alpha.size(0), alpha.size(0)).tril()[alpha.sort().indices]
        mask = torch.cat((mask, torch.zeros(1, locs.shape[-2])), dim=0)

        alpha_options = torch.einsum('ij, j->ij', mask, alpha)
        beta_options = torch.einsum('ij, j->ij', 1 - mask, beta)

        alpha_max = alpha_options.max(dim=-1).values
        result_options = alpha_max * w2_p__disc_q ** 2 + torch.einsum('j,ij->i', signature.probs, beta_options)

        return result_options.min(-1).values + w2_shift_locs

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
        locs_shift = torch.randn_like(signature.locs.detach()) * 0.01
        optimal_shift, losses = minimize_with_adam(
            param=locs_shift.requires_grad_(True),
            objective=fn_sq_w2_f_p__f_disc_q,
            **kwargs
        )
        w2 = fn_sq_w2_f_p__f_disc_q(optimal_shift).sqrt()
        w2_zero_shift = fn_sq_w2_f_p__f_disc_q().sqrt()
        if w2_zero_shift < w2:
            w2 = w2_zero_shift
            print('Optimal shift is zero!')
    else:
        w2 = fn_sq_w2_f_p__f_disc_q().sqrt()

    return w2
