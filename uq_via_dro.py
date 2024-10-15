import torch
import numpy as np
from scipy.optimize import linprog
from typing import Callable

import dynamics
import DistSignatures as ds
from regions import HyperRectangularVoronoiPartition


def get_lp2_norm_of_projection_matrix(signature: ds.DiscretizedMultivariateNormal,
                                      voronoi_partition: HyperRectangularVoronoiPartition):
    """
    Compute proj_{R_k}(c_i) for all c_i the signature locations and all regions R_k in the voronoi partition, and store
    in matrix with i-th row corresponding to c_i and k-th column corresponding to R_k.
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

def get_fn_bound_on_w2_fP_fdiscP(
                 signature: ds.DiscretizedMultivariateNormal,
                 f: dynamics.Dynamics,
                 budget: float) -> Callable:

    voronoi_partition = HyperRectangularVoronoiPartition(signature.locs)

    beta = f.bound_lp2_norm_difference(voronoi_partition).pow(2)
    l2_norm_proj_matrix = get_lp2_norm_of_projection_matrix(signature, voronoi_partition)

    def fn_bound_on_w2_fP_fdiscP(lambd: torch.Tensor):
        inner_sup = torch.max(beta - lambd * l2_norm_proj_matrix, dim=0).values
        return (lambd * budget ** 2 + torch.dot(signature.probs, inner_sup)).sqrt()

    return fn_bound_on_w2_fP_fdiscP


class BoundW2_fdiscP_vs_fdiscQ:
    def __init__(self, signature: ds.DiscretizedMultivariateNormal, f: dynamics.Dynamics, budget: float):
        self.signature_locs = signature.locs
        self.signature_probs = signature.probs

        f_signature_locs = f(signature.locs)
        self.F = torch.norm(f_signature_locs.unsqueeze(-3) - f_signature_locs.unsqueeze(-2), p=2, dim=-1).pow(2)
        self.C = torch.norm(self.signature_locs.unsqueeze(-3) - self.signature_locs.unsqueeze(-2), p=2, dim=-1).pow(2)
        self.budget = budget

    def solve_lin_problem(self):
        # Assume other variables are given and fixed
        n = self.signature_locs.shape[-2]
        # F = f.F(signatures).pow(2)
        # pi_q = initial_signature_probs
        # C = (signatures.view(1, -1) - signatures.view(-1, 1)).abs()
        # w = 2 * wasserstein_squared_zero

        # Reshape F, C, and Pi for linprog (they need to be 1D vectors)
        F_flat = self.F.flatten().numpy()
        C_flat = self.C.flatten().numpy()

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
        b_marg = self.signature_probs.numpy()

        # Wasserstein constraint: (C * Pi).sum() <= w
        A_ineq = np.array([C_flat])  # One inequality constraint for the Wasserstein bound
        b_ineq = [self.budget **  2]

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
            return (self.F * Pi_optimized).sum().sqrt()
        else:
            raise ValueError(f"Optimization failed: {result.message}")