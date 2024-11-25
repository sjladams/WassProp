import torch
from typing import Callable

import dynamics
import discretize_distributions as ds
from regions import HyperRectangularVoronoiPartition

from bound import local_ibp_sq_norm_fx_fc, get_norm_of_proj_matrix, global_ibp_sq_norm_fx_fc, global_lbp_sq_norm_fx_fc

from optimize import minimize_with_adam


# ----- W_2(f#p, f#disc#p) -----
def get_fn_sq_w2_f_p__f_disc_p(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        center_ball: ds.MultivariateNormal) -> Callable:

    voronoi_partition = HyperRectangularVoronoiPartition(signature.locs)
    alphas = global_lbp_sq_norm_fx_fc(f, voronoi_partition)
    locs = signature.locs
    means = center_ball.mean
    sigmas = center_ball._sqrt_diag_covariance_matrix
    lower = voronoi_partition.lower
    upper = voronoi_partition.upper

    upper[torch.isposinf(upper)] = 1e4 # large value
    lower[torch.isneginf(lower)] = -1e4

    def fn_sq_w2_f_p__f_disc_p():

        scaled_lower = (lower - means) / (2**0.5 * sigmas)
        scaled_upper = (upper - means) / (2 ** 0.5 * sigmas)

        erf_lower = torch.special.erf(scaled_lower)
        erf_upper = torch.special.erf(scaled_upper)

        common_factor = 0.5 * ((locs - means) ** 2 + sigmas ** 2)

        factor_exp_upper = (1 / (2 * torch.pi) ** 0.5) * sigmas * (upper - 2 * locs + means) * torch.exp(-scaled_upper ** 2)
        factor_exp_lower = (1 / (2 * torch.pi) ** 0.5) * sigmas * (lower - 2 * locs + means) * torch.exp(
            -scaled_lower ** 2)

        #Using Mathematica
        integral_per_dim = common_factor * (erf_upper - erf_lower) - factor_exp_upper + factor_exp_lower
        integral = torch.prod(integral_per_dim, dim=1)

        return torch.dot(alphas, integral)

    return fn_sq_w2_f_p__f_disc_p

def compute_w2_f_p__f_disc_p(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float,
        center_ball: ds.MultivariateNormal,
        **kwargs):

    fn_sq_w2_f_p__f_disc_p = get_fn_sq_w2_f_p__f_disc_p(signature, f, center_ball)

    return fn_sq_w2_f_p__f_disc_p().sqrt().detach()


# ----- W_2(f#p, f#disc#q) for Independent Coupling approach -----
def get_fn_sq_w2_f_p__f_disc_q_independent_coupling(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float) -> Callable:

    voronoi_partition = HyperRectangularVoronoiPartition(signature.locs)

    w2_p__disc_q = w2_q__disc_q + w2_p__q

    alpha = global_lbp_sq_norm_fx_fc(f, voronoi_partition)

    def fn_sq_w2_f_p__f_disc_q_independent_coupling(lambd: torch.Tensor):

        locs = signature.locs.detach()
        probs = signature.probs
        budget_sq = (w2_p__disc_q ** 2).detach()

        v = lambd * locs - ((probs * alpha).unsqueeze(1) * locs).sum(dim=0, keepdim=True)
        coeff_v = 1 / (lambd - torch.dot(probs, alpha))

        c__transpose__c = torch.sum(locs ** 2, dim=1)
        sum_pi_alpha_c__transpose__c = torch.sum(probs * alpha * c__transpose__c)

        quadrat_sol = coeff_v * (v ** 2).sum(dim=1) - lambd * (locs ** 2).sum(dim=1) + sum_pi_alpha_c__transpose__c

        result = lambd * budget_sq + torch.dot(probs, quadrat_sol)

        return result

    return fn_sq_w2_f_p__f_disc_q_independent_coupling

def compute_w2_f_p__f_disc_q_independent_coupling(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float,
        center_ball: ds.MultivariateNormal,
        **kwargs):

    if w2_p__q == 0.:
        return compute_w2_f_p__f_disc_p(signature, f, w2_q__disc_q, w2_p__q, center_ball, ** kwargs)

    fn_sq_w2_f_p__f_disc_q = get_fn_sq_w2_f_p__f_disc_q_independent_coupling(
        signature, f, w2_q__disc_q, w2_p__q)

    voronoi_partition = HyperRectangularVoronoiPartition(signature.locs)
    alpha = global_lbp_sq_norm_fx_fc(f, voronoi_partition)
    avg_alpha = torch.dot(alpha, signature.probs).detach() + 1e-3

    optimized_lambda, losses = minimize_with_adam(
        param=torch.tensor(15., requires_grad=True),
        objective=fn_sq_w2_f_p__f_disc_q,
        non_negative_constraint=True,
        min_value=avg_alpha,
        **kwargs
    )

    return fn_sq_w2_f_p__f_disc_q(optimized_lambda).sqrt().detach()


# ----- W_2(f#p, f#disc#q) for Lagragian Duality approach -----
def get_fn_sq_w2_f_p__f_disc_q_local_linear_or_constant(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float)-> Callable:

    voronoi_partition = HyperRectangularVoronoiPartition(signature.locs)

    w2_p__disc_q = w2_q__disc_q + w2_p__q

    alpha = global_lbp_sq_norm_fx_fc(f, voronoi_partition)
    beta  = global_ibp_sq_norm_fx_fc(f, voronoi_partition).upper.squeeze(-1)

    mask = torch.ones(alpha.size(0), alpha.size(0)).tril()[alpha.sort().indices]
    mask = torch.cat((mask, torch.zeros(1, voronoi_partition.num_locs)), dim=0)

    alpha_options = torch.einsum('ij, j->ij', mask, alpha)
    beta_options = torch.einsum('ij, j->ij', 1 - mask, beta)

    def fn_sq_w2_f_p__f_disc_q_local_linear_or_constant():
        alpha_max = alpha_options.max(dim=-1).values
        result_options = alpha_max * w2_p__disc_q ** 2 + torch.einsum('j,ij->i', signature.probs, beta_options)

        return result_options.min()

    return fn_sq_w2_f_p__f_disc_q_local_linear_or_constant

def compute_w2_f_p__f_disc_q_local_linear_or_constant(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float,
        center_ball: ds.MultivariateNormal,
        **kwargs):

    if w2_p__q == 0.:
        return compute_w2_f_p__f_disc_p(signature, f, w2_q__disc_q, w2_p__q, center_ball, ** kwargs)

    fn_sq_w2_f_p__f_disc_q = get_fn_sq_w2_f_p__f_disc_q_local_linear_or_constant(signature, f, w2_q__disc_q, w2_p__q)

    return fn_sq_w2_f_p__f_disc_q().sqrt().detach()