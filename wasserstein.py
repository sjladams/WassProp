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
        non_negative_constraint=True,
        **kwargs)

    return fn_sq_w2_f_p__f_disc_p(optimized_lambda).sqrt().detach()


# ----- W_2(f#p, f#disc#q) for Independent Coupling approach -----
def get_fn_sq_w2_f_p__f_disc_q_independent_coupling(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float) -> Callable:

    voronoi_partition = HyperRectangularVoronoiPartition(signature.locs)

    sq_norm_fx_fc = local_ibp_sq_norm_fx_fc(f, voronoi_partition).upper.squeeze(-1)
    averaged_sq_norm_fx_fc = torch.einsum('jl,j->l', sq_norm_fx_fc, signature.probs)

    sq_norm_proj_matrix = get_norm_of_proj_matrix(voronoi_partition).pow(2)

    w2_p__disc_q = w2_q__disc_q + w2_p__q

    def fn_sq_w2_f_p__f_disc_q_independent_coupling(lambd: torch.Tensor):
        inner_sup = torch.max(averaged_sq_norm_fx_fc - lambd * sq_norm_proj_matrix, dim=-1).values
        return lambd * w2_p__disc_q ** 2 + torch.einsum('m,m->', signature.probs, inner_sup)

    return fn_sq_w2_f_p__f_disc_q_independent_coupling


def compute_w2_f_p__f_disc_q_independent_coupling(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float,
        **kwargs):

    fn_sq_w2_f_p__f_disc_q = get_fn_sq_w2_f_p__f_disc_q_independent_coupling(
        signature, f, w2_q__disc_q, w2_p__q)

    optimized_lambda, losses = minimize_with_adam(
        param=torch.tensor(0.1, requires_grad=True),
        objective=fn_sq_w2_f_p__f_disc_q,
        non_negative_constraint=True,
        **kwargs
    )

    return fn_sq_w2_f_p__f_disc_q(optimized_lambda).sqrt().detach()


# ----- W_2(f#p, f#disc#q) for Lagrangian Duality approach -----
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
        **kwargs):

    fn_sq_w2_f_p__f_disc_q = get_fn_sq_w2_f_p__f_disc_q_local_linear_or_constant(signature, f, w2_q__disc_q, w2_p__q)

    return fn_sq_w2_f_p__f_disc_q().sqrt().detach()
