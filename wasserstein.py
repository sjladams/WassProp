import torch
import math
import numpy as np
from scipy.optimize import linprog
from typing import Callable

import dynamics
import discretize_distributions as ds
from regions import HyperRectangularVoronoiPartition

from optimization_utils import minimize_with_adam


def get_lp2_norm_of_proj_matrix(signature: ds.DiscretizedMultivariateNormal,
                                voronoi_partition: HyperRectangularVoronoiPartition):
    """
    Compute proj_{R_k}(c_i) for all c_i the signature locations and all regions R_k in the voronoi partition, and store
    in matrix with i-th row corresponding to c_i and k-th column corresponding to R_k.

    CONVENTION: indexing regions over columns and centers over rows

    :param signature:
    :param voronoi_partition:
    :return:
    """
    locs_expanded = signature.locs.unsqueeze(-3)
    lower_expanded = voronoi_partition.lower.unsqueeze(-2)
    upper_expanded = voronoi_partition.upper.unsqueeze(-2)

    # Compute the projections for all combinations
    below_lower = torch.where(locs_expanded < lower_expanded, lower_expanded - locs_expanded,
                              torch.zeros_like(locs_expanded))
    above_upper = torch.where(locs_expanded > upper_expanded, locs_expanded - upper_expanded,
                              torch.zeros_like(locs_expanded))

    # Calculate the projection, summing both below and above cases
    proj_matrix = below_lower + above_upper

    return torch.norm(proj_matrix, dim=-1, p=2)


# ----- W_2(f#p, f#disc#p) -----
def get_fn_sq_w2_f_p__f_disc_p(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float) -> Callable:

    voronoi_partition = HyperRectangularVoronoiPartition(signature.locs_inner, signature.loc_shell, signature.shell)

    lp2_norm_diff_sq = f.bound_lp2_norm_difference(voronoi_partition).diag().pow(2)
    lp2_norm_proj_matrix_sq = get_lp2_norm_of_proj_matrix(signature, voronoi_partition).pow(2)

    w2_p__disc_q = w2_q__disc_q + w2_p__q

    def fn_sq_w2_f_p__f_disc_p(lambd: torch.Tensor):
        inner_sup = torch.max(lp2_norm_diff_sq - lambd * lp2_norm_proj_matrix_sq, dim=-1).values
        return lambd * w2_p__disc_q ** 2 + torch.einsum('m,m->', signature.probs, inner_sup)

    return fn_sq_w2_f_p__f_disc_p

def compute_w2_f_p__f_disc_p(signature: ds.DiscretizedMultivariateNormal,
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

    f_signature_locs = f(signature.locs)
    F_sq = torch.norm(f_signature_locs.unsqueeze(-3) - f_signature_locs.unsqueeze(-2), p=2, dim=-1).pow(2)
    lp2_norm_proj_matrix_sq = get_lp2_norm_of_proj_matrix(signature, voronoi_partition).pow(2)

    w2_p__disc_q = w2_q__disc_q + w2_p__q

    def fn_sq_w2_f_disc_p__f_disc_q(lambd: torch.Tensor):
        inner_sup = torch.max(F_sq - lambd * lp2_norm_proj_matrix_sq, dim=-1).values
        return lambd * w2_p__disc_q ** 2 + torch.einsum('m,m->', signature.probs, inner_sup)

    return fn_sq_w2_f_disc_p__f_disc_q


def compute_w2_f_disc_p__f_disc_q(signature: ds.DiscretizedMultivariateNormal,
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
    else:
        raise ValueError("Invalid budget type. Choose either 'w2_p__disc_q' or 'w2_disc_p__disc_q'.")



# ----- W_2(f#p, f#disc#q) for independent coupling -----
def get_fn_sq_w2_f_p__f_disc_q_independent_coupling(
        signature: ds.DiscretizedMultivariateNormal,
        f: dynamics.Dynamics,
        w2_q__disc_q: float,
        w2_p__q: float) -> Callable:

    voronoi_partition = HyperRectangularVoronoiPartition(signature.locs_inner, signature.loc_shell, signature.shell)

    lp2_norm_diff_matrix_sq = f.bound_lp2_norm_difference(voronoi_partition).pow(2)
    average_lp2_norm_diff_sq = torch.einsum('jl,j->l', lp2_norm_diff_matrix_sq, signature.probs)

    lp2_norm_proj_matrix_sq = get_lp2_norm_of_proj_matrix(signature, voronoi_partition).pow(2)

    w2_p__disc_q = w2_q__disc_q + w2_p__q

    def fn_sq_w2_f_p__f_disc_q_independent_coupling(lambd: torch.Tensor):
        inner_sup = torch.max(average_lp2_norm_diff_sq - lambd * lp2_norm_proj_matrix_sq, dim=-1).values
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


# ----- W_2(f#p, f#disc#q) - New method -----
def get_fn_sq_w2_f_p__f_disc_q_together(signature: ds.DiscretizedMultivariateNormal,
                                                 f: dynamics.Dynamics,
                                                 w2_q__disc_q: float,
                                                 w2_p__q: float) -> Callable:

    voronoi_partition = HyperRectangularVoronoiPartition(signature.locs_inner, signature.loc_shell, signature.shell)

    if w2_p__q == 0.:
        F_sq = torch.zeros(voronoi_partition.num_locs, voronoi_partition.num_locs)

        factor = 1
    else:
        f_signature_locs = f(signature.locs)
        F_sq = torch.norm(f_signature_locs.unsqueeze(-3) - f_signature_locs.unsqueeze(-2), p=2, dim=-1).pow(2)

        factor = math.sqrt(2)

    lp2_norm_diff_vec_sq = f.bound_lp2_norm_difference(voronoi_partition).diag().pow(2)
    lp2_norm_proj_matrix_sq = get_lp2_norm_of_proj_matrix(signature, voronoi_partition).pow(2)

    w2_p__disc_q = w2_q__disc_q + w2_p__q

    def fn_sq_w2_f_p__f_disc_q_together(lambd: torch.Tensor):
        inner_sup = torch.max(lp2_norm_diff_vec_sq + F_sq - lambd * lp2_norm_proj_matrix_sq, dim=-1).values
        return factor * (lambd * w2_p__disc_q ** 2 + torch.einsum('m,m->', signature.probs, inner_sup))

    return fn_sq_w2_f_p__f_disc_q_together

def compute_w2_f_p__f_disc_q_together(signature: ds.DiscretizedMultivariateNormal,
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