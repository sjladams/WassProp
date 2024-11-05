import torch
import math
import numpy as np
from scipy.optimize import linprog
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

    voronoi_partition = HyperRectangularVoronoiPartition(signature.locs_inner, signature.loc_shell, signature.shell)

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


# ----- W_2(f#disc#p, f#disc#q) -----
def get_fn_sq_w2_f_disc_p__f_disc_q(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float) -> Callable:

    voronoi_partition = HyperRectangularVoronoiPartition(signature.locs_inner, signature.loc_shell, signature.shell)

    fc = f(signature.locs)
    sq_norm_fc_fc = torch.norm(fc.unsqueeze(-3) - fc.unsqueeze(-2), p=2, dim=-1).pow(2)
    sq_norm_proj_matrix = get_norm_of_proj_matrix(voronoi_partition).pow(2)

    w2_p__disc_q = w2_q__disc_q + w2_p__q

    def fn_sq_w2_f_disc_p__f_disc_q(lambd: torch.Tensor):
        inner_sup = torch.max(sq_norm_fc_fc - lambd * sq_norm_proj_matrix, dim=-1).values
        return lambd * w2_p__disc_q ** 2 + torch.einsum('m,m->', signature.probs, inner_sup)

    return fn_sq_w2_f_disc_p__f_disc_q


def compute_w2_f_disc_p__f_disc_q(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float,
        budget_type: str = 'w2_p__disc_q',
        **kwargs):

    if w2_p__q == 0.:
        return torch.tensor(0.)
    elif budget_type == 'w2_p__disc_q':
        fn_sq_w2_f_disc_p__f_disc_q = get_fn_sq_w2_f_disc_p__f_disc_q(signature, f, w2_q__disc_q, w2_p__q)

        optimized_lambda, losses = minimize_with_adam(
            param=torch.tensor(0.1, requires_grad=True),
            objective=fn_sq_w2_f_disc_p__f_disc_q,
            non_negative_constraint=True,
            **kwargs)

        return fn_sq_w2_f_disc_p__f_disc_q(optimized_lambda).sqrt().detach()
    elif budget_type == 'w2_disc_p__disc_q':
        return compute_w2_f_disc_p__f_disc_q_via_linprog(signature, f, w2_q__disc_q, w2_p__q)
    else:
        raise ValueError("Invalid budget type. Choose either 'w2_p__disc_q' or 'w2_disc_p__disc_q'.")


def compute_w2_f_disc_p__f_disc_q_via_linprog(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float):
    w2_disc_p__disc_q = 2 * (w2_q__disc_q + w2_p__q)

    f_signature_locs = f(signature.locs)
    F = torch.norm(f_signature_locs.unsqueeze(-3) - f_signature_locs.unsqueeze(-2), p=2, dim=-1).pow(2)
    C = torch.norm(signature.locs.unsqueeze(-3) - signature.locs.unsqueeze(-2), p=2, dim=-1).pow(2)

    # Assume other variables are given and fixed
    n = signature.locs.shape[-2]

    # Reshape F, C, and Pi for linprog (they need to be 1D vectors)
    F_flat = F.flatten().numpy()
    C_flat = C.flatten().numpy()

    # Objective function is to maximize F * Pi, which is the same as minimizing -(F * Pi)
    c = -F_flat  # Minimizing -F is the same as maximizing F

    # Constraints:
    # Simplex constraint (Pi.sum() == 1): equality constraint
    A_eq = np.ones((1, n * n))  # Sum of all elements in Pi should be 1
    b_eq = [1]

    # Marginal equality constraint: Pi.sum(dim=0) == pi_q
    A_marg = np.zeros((n, n * n))
    for i in range(n):
        A_marg[i, i::n] = 1  # Select rows corresponding to each column sum
    b_marg = signature.probs.numpy()

    # Wasserstein constraint: (C * Pi).sum() <= w
    A_ineq = np.array([C_flat])  # One inequality constraint for the Wasserstein bound
    b_ineq = [w2_disc_p__disc_q ** 2]

    # Combine constraints
    A_eq_combined = np.vstack([A_eq, A_marg])  # Combine the equality constraints
    b_eq_combined = np.hstack([b_eq, b_marg])

    # Bounds for each element of Pi: 0 <= Pi <= infinity (non-negative)
    bounds = [(0, 1)] * (n * n)

    # Solve the linear program
    result = linprog(c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq_combined, b_eq=b_eq_combined, bounds=bounds,
                     method='highs')

    if result.success:
        Pi_optimized = result.x.reshape(n, n)
        return (F * Pi_optimized).sum().sqrt()
    else:
        raise ValueError(f"Optimization failed: {result.message}")

# ----- W_2(f#p, f#disc#q) for independent coupling -----
def get_fn_sq_w2_f_p__f_disc_q_independent_coupling(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float) -> Callable:

    voronoi_partition = HyperRectangularVoronoiPartition(signature.locs_inner, signature.loc_shell, signature.shell)

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


# ----- W_2(f#p, f#disc#q) - Together Method -----
def get_fn_sq_w2_f_p__f_disc_q_together(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float) -> Callable:

    voronoi_partition = HyperRectangularVoronoiPartition(signature.locs_inner, signature.loc_shell, signature.shell)

    if w2_p__q == 0.:
        sq_norm_fc_fc = torch.zeros(voronoi_partition.num_locs, voronoi_partition.num_locs)

        factor = 1
    else:
        fc = f(signature.locs)
        sq_norm_fc_fc = torch.norm(fc.unsqueeze(-3) - fc.unsqueeze(-2), p=2, dim=-1).pow(2)

        factor = math.sqrt(2)

    sq_norm_fx_fc = local_ibp_sq_norm_fx_fc(f, voronoi_partition).upper.squeeze(-1).diag()
    sq_norm_proj_matrix = get_norm_of_proj_matrix(voronoi_partition).pow(2)

    w2_p__disc_q = w2_q__disc_q + w2_p__q

    def fn_sq_w2_f_p__f_disc_q_together(lambd: torch.Tensor):
        inner_sup = torch.max(sq_norm_fx_fc + sq_norm_fc_fc - lambd * sq_norm_proj_matrix, dim=-1).values
        return factor * (lambd * w2_p__disc_q ** 2 + torch.einsum('m,m->', signature.probs, inner_sup))

    return fn_sq_w2_f_p__f_disc_q_together

def compute_w2_f_p__f_disc_q_together(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float,
        **kwargs):
    fn_sq_w2_f_p__f_disc_q = get_fn_sq_w2_f_p__f_disc_q_together(signature, f, w2_q__disc_q, w2_p__q)

    optimized_lambda, losses = minimize_with_adam(
        param=torch.tensor(0.1, requires_grad=True),
        objective=fn_sq_w2_f_p__f_disc_q,
        non_negative_constraint=True,
        **kwargs)

    return fn_sq_w2_f_p__f_disc_q(optimized_lambda).sqrt().detach()


# ----- W_2(f#p, f#disc#q) - Local Linearization Method -----
def get_fn_sq_w2_f_p__f_disc_q_local_linear(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float)-> Callable:

    voronoi_partition = HyperRectangularVoronoiPartition(signature.locs_inner, signature.loc_shell, signature.shell)

    alpha = global_lbp_sq_norm_fx_fc(f, voronoi_partition)
    alpha_max = alpha[signature.probs > 0.].max(dim=-1).values

    w2_p__disc_q = w2_q__disc_q + w2_p__q

    def fn_sq_w2_f_p__f_disc_q_local_linear():
        return alpha_max * w2_p__disc_q ** 2

    return fn_sq_w2_f_p__f_disc_q_local_linear


def compute_w2_f_p__f_disc_q_local_linear(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float,
        **kwargs):
    fn_sq_w2_f_p__f_disc_q = get_fn_sq_w2_f_p__f_disc_q_local_linear(signature, f, w2_q__disc_q, w2_p__q)
    return fn_sq_w2_f_p__f_disc_q().sqrt().detach()

def get_fn_sq_w2_f_p__f_disc_q_local_constant(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float)-> Callable:

    voronoi_partition = HyperRectangularVoronoiPartition(signature.locs_inner, signature.loc_shell, signature.shell)

    beta  = global_ibp_sq_norm_fx_fc(f, voronoi_partition).upper.squeeze(-1)

    def fn_sq_w2_f_p__f_disc_q_local_constant():
        return  torch.einsum('i,i->', signature.probs, beta)

    return fn_sq_w2_f_p__f_disc_q_local_constant


def compute_w2_f_p__f_disc_q_local_constant(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float,
        **kwargs):
    fn_sq_w2_f_p__f_disc_q = get_fn_sq_w2_f_p__f_disc_q_local_constant(signature, f, w2_q__disc_q, w2_p__q)
    return fn_sq_w2_f_p__f_disc_q().sqrt().detach()

