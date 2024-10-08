import torch
import numpy as np
from bound_propagation import HyperRectangle
from scipy.optimize import linprog

import dynamics
import DistSignatures as ds


class BoundW2_f_push_P_vs_f_push_SignatureP:
    def __init__(self, signature: ds.DiscretizedMultivariateNormal, f: dynamics.Dynamics, regions: HyperRectangle,
                 budget: float):
        self.signature = signature
        self.beta = self.get_beta(f, regions)
        self.lp2_norm_proj_matrix = self.get_lp2_norm_of_projection_matrix(signature, regions)
        self.budget = budget

    @staticmethod
    def get_beta(f: dynamics.Dynamics, regions: HyperRectangle):
        return f.interval_approximation(regions).pow(2)

    @staticmethod
    def get_lp2_norm_of_projection_matrix(signature: ds.DiscretizedMultivariateNormal, regions: HyperRectangle):
        #\todo please double check the procedure below
        print("get_projection_matrix only works for axis-aligned regions!")

        locs_expanded = signature.locs.unsqueeze(-2)
        lower_expanded = regions.lower.unsqueeze(-3)
        upper_expanded = regions.upper.unsqueeze(-3)

        # Compute the projections for all combinations
        below_lower = torch.where(locs_expanded < lower_expanded, lower_expanded - locs_expanded,
                                  torch.zeros_like(locs_expanded))
        above_upper = torch.where(locs_expanded > upper_expanded, locs_expanded - upper_expanded,
                                  torch.zeros_like(locs_expanded))

        # Calculate the projection, summing both below and above cases
        proj_matrix = below_lower + above_upper

        # Compute the squared projections and sum over the dimensions
        lp2_norm_proj_matrix = torch.sum(proj_matrix ** 2, dim=-1)

        return lp2_norm_proj_matrix

    def get_objective(self):
        def objective(lambd: torch.Tensor):
            value_matrix = self.beta - lambd * self.lp2_norm_proj_matrix.unsqueeze(-1)

            # Take the max over the computed value_matrix
            max_values = value_matrix.max(0).values.squeeze() #TODO: CHECK SQUEEZE

            # Compute the outer_sum using vectorized operations
            outer_sum = torch.sum(self.signature.probs * max_values)
            outer_sum += lambd * self.budget ** 2

            return torch.sqrt(outer_sum)

        return objective


class BoundW2_f_push_SignatureP_vs_f_push_SignatureQ:
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