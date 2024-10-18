import ot
import torch
import matplotlib.pyplot as plt
from copy import copy
from typing import Union, List

import discretize_distributions as ds
import GMMWas
import wasserstein
from dynamics import Dynamics


def multi_step(dynamics: Dynamics,
               loc_noise_dist: torch.Tensor,
               covariance_noise_dist: torch.Tensor,
               loc_initial_dist: torch.Tensor,
               covariance_initial_dist: torch.Tensor,
               num_samples: int,
               num_time_steps: int,
               num_signature_points: int,
               run_type1: bool = False,
               plot: bool = False):

    # Initialize System Dynamics
    print(f"Global Lipschitz constant of f: {dynamics.global_lipschitz}")
    noise_distribution = ds.MultivariateNormal(loc=loc_noise_dist, covariance_matrix=covariance_noise_dist)

    if dynamics.num_dims == 1 and plot:
        # Plot dynamics
        x = torch.linspace(start=-5, end=5, steps=500).view(-1, 1)
        y = dynamics(x)

        fig_dynamics = plt.figure()
        plt.plot(x, y)
        plt.title("Dynamics f(x)")
        plt.show()

    # Initialize state distributions
    q0 = ds.MultivariateNormal(loc=loc_initial_dist, covariance_matrix=covariance_initial_dist)
    q0_samples = q0.sample(torch.Size((num_samples,)))
    p0_samples = q0_samples.clone()

    # store wasserstein error bounds
    w2_empirical = torch.zeros(num_time_steps+1)
    w2_global_lipschitz = torch.zeros(num_time_steps+1)
    w2_type1 = torch.zeros(num_time_steps+1)
    w2_type2 = torch.zeros(num_time_steps+1)

    # Propagate the system
    for k in range(num_time_steps):
        ### Generate the signature of the approximate state distribution at time k:
        sign_q0 = ds.discretization_generator(dist=q0, compute_w2=True, nr_signature_points=num_signature_points)

        print(f"W_2(q_{k}, Delta_C#q_{k}) = {sign_q0.w2:.4f}")

        ### Propagate the (approximate) state distribution over the dynamics
        q1 = ds.MixtureMultivariateNormal(
            mixture_distribution=torch.distributions.Categorical(
                probs=sign_q0.probs),
            component_distribution=ds.MultivariateNormal(
                loc=dynamics(sign_q0.locs) + noise_distribution.loc,
                covariance_matrix=noise_distribution.covariance_matrix
            ))
        q1_compr = copy(q1)
        q1_compr.compress(n_max=num_signature_points)
        w2_compr = GMMWas.w2(q1_compr, q1)

        q1_compr_samples = q1_compr.sample(torch.Size((num_samples,)))
        p1_samples = dynamics(p0_samples) + noise_distribution.sample(torch.Size((num_samples,)))

        #### Compute W_2(p_{k+1}, q_{k+1}) = W_2(f#p_k, f#\Delta_C#q_k)
        ### Empirical Approximate
        w2_empirical[k+1] = ot.solve_sample(p1_samples.view(-1, dynamics.num_dims),
                                            q1_compr_samples.view(-1, dynamics.num_dims)
                                            ).value.sqrt()

        ### Global Lipschitz
        w2_global_lipschitz[k+1] = dynamics.global_lipschitz * (sign_q0.w2 + w2_compr + w2_global_lipschitz[k])

        ### Our Method
        ## Type 1: Budget Term 2 = W_2(disc # p, disc # q)
        if run_type1:
            # Term 1: bound W_2(f#p_k, f#\Delta_C#p_k)
            w2_term1_type1 = wasserstein.compute_bound_w2_f_p__f_disc_p(
                sign_q0, dynamics, budget=sign_q0.w2 + w2_compr + w2_type1[k])

            # Term 2: bound W_2(f#\Delta#p_k, f#\Delta#q_k)
            if k == 0:
                w2_term2_type1 = 0.
            else:
                w2_term2_type1 = wasserstein.compute_bound_w2_f_disc_p__f_disc_q(
                    sign_q0, dynamics, budget=2 * (sign_q0.w2 + w2_compr + w2_type1[k]), budget_type='w2_disc_p__disc_q')

            w2_type1[k + 1] = w2_term1_type1 + w2_term2_type1


        ### Type 2: Budget Term 2 = W_2(p, disc # q)
        ## Term 1: bound W_2(f#p_k, f#\Delta_C#p_k)
        w2_term1_type2 = wasserstein.compute_bound_w2_f_p__f_disc_p(
            sign_q0, dynamics, budget=sign_q0.w2 + w2_compr + w2_type2[k])

        ## Term 2: bound W_2(f#\Delta#p_k, f#\Delta#q_k)
        if k == 0:
            w2_term2_type2 = 0.
        else:
            w2_term2_type2 = wasserstein.compute_bound_w2_f_disc_p__f_disc_q(
                sign_q0, dynamics, budget= (sign_q0.w2 + w2_compr + w2_type2[k]), budget_type='w2_p__disc_q')

        w2_type2[k + 1] = w2_term1_type2 + w2_term2_type2

        print(f"Bounds on W_2(W_2(p_{k+1}, q_{k+1})) via:\n"
              f"\t Global Lipschits: {w2_global_lipschitz[k+1]:.4f}\n"
              f"\t Empirical: {w2_empirical[k+1]:.4f}\n")
        if run_type1:
            print(f"\t Own Type 1: {w2_type1[k + 1]:.4f}\n"
                  f"\t\t\t Term 1: {w2_term1_type1:.4f}\n"
                  f"\t\t\t Term 2: {w2_term2_type1:.4f}\n")
        print(f"\t Own Type 2: {w2_type2[k + 1]:.4f}\n"
              f"\t\t\t Term 1: {w2_term1_type2:.4f}\n"
              f"\t\t\t Term 2: {w2_term2_type2:.4f}")

        if dynamics.num_dims == 1 and plot:
            fig_propagation = plt.figure()
            plt.hist(p0_samples.squeeze(), alpha=0.5, label=f'k={k}', bins=100, density=True)
            plt.hist(p1_samples.squeeze(), alpha=0.5, label=f'k={k+1}', bins=100, density=True)
            plt.legend()
            plt.title(f"Histograms of p_{k} and p_{k+1}")
            plt.show()

            fig_signature = plt.figure()
            plt.bar(sign_q0.locs.squeeze(), sign_q0.probs, width=0.1)
            plt.hist(q0_samples, alpha=0.5, label=f'k={k}', bins=100, density=True)
            plt.title(f"Signature of q_{k} and Histogram of q_{k}")
            plt.show()

            fig_approximation_error = plt.figure()
            plt.hist(p1_samples, alpha=0.5, label=f'p_{k}', bins=100, density=True)
            plt.hist(q1_compr_samples, alpha=0.5, label=f'q_{k}', bins=100, density=True)
            plt.legend()
            plt.title("Histograms q1 vs p1")
            plt.show()


        # Overwrite for next iteration
        q0 = q1_compr
        q0_samples = q1_compr_samples
        p0_samples = p1_samples

    tag = f"{dynamics.__class__.__name__} (Lipschitz={dynamics.global_lipschitz:.2f}, |C|={num_signature_points})"
    w2_bounds = {'emp': w2_empirical, 'gl': w2_global_lipschitz, 'type2': w2_type2}
    if run_type1:
        w2_bounds['type1'] = w2_type1
    return w2_bounds, tag


def single_step(dynamics: Dynamics,
                loc_noise_dist: torch.Tensor,
                covariance_noise_dist: torch.Tensor,
                loc_initial_dist: torch.Tensor,
                covariance_initial_dist: torch.Tensor,
                num_samples: int,
                num_signature_points: int,
                initial_budget_options: Union[List, float],
                run_type1: bool = False,
                plot: bool = False):

    if isinstance(initial_budget_options, float):
        initial_budget_options = [initial_budget_options]

    # Initialize System Dynamics
    print(f"Global Lipschitz constant of f: {dynamics.global_lipschitz}")
    noise_distribution = ds.MultivariateNormal(loc=loc_noise_dist, covariance_matrix=covariance_noise_dist)

    if dynamics.num_dims == 1 and plot:
        # Plot dynamics
        x = torch.linspace(start=-5, end=5, steps=500).view(-1, 1)
        y = dynamics(x)

        fig_dynamics = plt.figure()
        plt.plot(x, y)
        plt.title("Dynamics f(x)")
        plt.show()

    # Initialize state distributions
    q0 = ds.MultivariateNormal(loc=loc_initial_dist, covariance_matrix=covariance_initial_dist)
    q0_samples = q0.sample(torch.Size((num_samples,)))

    # store wasserstein error bounds
    w2_global_lipschitz = torch.zeros(len(initial_budget_options))
    w2_type1 = torch.zeros(len(initial_budget_options))
    w2_type2 = torch.zeros(len(initial_budget_options))

    # Propagate the system
    sign_q0 = ds.discretization_generator(dist=q0, compute_w2=True, nr_signature_points=num_signature_points)

    print(f"W_2(q_0, Delta_C#q_0) = {sign_q0.w2:.4f}")

    ### Propagate the (approximate) state distribution over the dynamics
    q1 = ds.MixtureMultivariateNormal(
        mixture_distribution=torch.distributions.Categorical(
            probs=sign_q0.probs),
        component_distribution=ds.MultivariateNormal(
            loc=dynamics(sign_q0.locs) + noise_distribution.loc,
            covariance_matrix=noise_distribution.covariance_matrix
        ))

    q1_samples = q1.sample(torch.Size((num_samples,)))
    f_q0_samples = dynamics(q0_samples) + noise_distribution.sample(torch.Size((num_samples,)))

    if dynamics.num_dims == 1 and plot:
        fig_propagation = plt.figure()
        plt.hist(q0_samples.squeeze(), alpha=0.5, label='q', bins=100, density=True)
        plt.hist(f_q0_samples.squeeze(), alpha=0.5, label='f#q', bins=100, density=True)
        plt.legend()
        plt.title(f"Histograms of q and f#q")
        plt.show()

        fig_signature = plt.figure()
        plt.bar(sign_q0.locs.squeeze(), sign_q0.probs, width=0.1)
        plt.hist(q0_samples, alpha=0.5, label='q', bins=100, density=True)
        plt.title(f"Signature of q and Histogram of q")
        plt.show()

    #### Compute W_2(p_1, q_1) = W_2(f#p_k, f#\Delta_C#q_k)
    for idx, initial_budget in enumerate(initial_budget_options):
        ### Global Lipschitz
        w2_global_lipschitz[idx] = dynamics.global_lipschitz * (sign_q0.w2 + initial_budget)

        ### Our Method
        ## Type 1: Budget Term 2 = W_2(disc # p, disc # q)
        if run_type1:
            # Term 1: bound W_2(f#p_k, f#\Delta_C#p_k)
            w2_term1_type1 = wasserstein.compute_bound_w2_f_p__f_disc_p(
                sign_q0, dynamics, budget=sign_q0.w2 + initial_budget)

            # Term 2: bound W_2(f#\Delta#p_k, f#\Delta#q_k)
            if initial_budget == 0.:
                w2_term2_type1 = 0.
            else:
                w2_term2_type1 = wasserstein.compute_bound_w2_f_disc_p__f_disc_q(
                    sign_q0, dynamics, budget=2 * (sign_q0.w2 + initial_budget), budget_type='w2_disc_p__disc_q')

            w2_type1[idx] = w2_term1_type1 + w2_term2_type1


        ### Type 2: Budget Term 2 = W_2(p, disc # q)
        ## Term 1: bound W_2(f#p_k, f#\Delta_C#p_k)
        w2_term1_type2 = wasserstein.compute_bound_w2_f_p__f_disc_p(
            sign_q0, dynamics, budget=sign_q0.w2 + initial_budget)

        ## Term 2: bound W_2(f#\Delta#p_k, f#\Delta#q_k)
        if initial_budget == 0.:
            w2_term2_type2 = 0.
        else:
            w2_term2_type2 = wasserstein.compute_bound_w2_f_disc_p__f_disc_q(
                sign_q0, dynamics, budget= (sign_q0.w2 + initial_budget), budget_type='w2_p__disc_q')

        w2_type2[idx] = w2_term1_type2 + w2_term2_type2

        print(f"Bounds on W_2(W_2(f#p, f#disc#q)) for W_2(p,q) = {initial_budget} via:\n"
              f"\t Global Lipschits: {w2_global_lipschitz[idx]:.4f}\n")
        if run_type1:
            print(f"\t Own Type 1: {w2_type1[idx]:.4f}\n"
                  f"\t\t\t Term 1: {w2_term1_type1:.4f}\n"
                  f"\t\t\t Term 2: {w2_term2_type1:.4f}\n")
        print(f"\t Own Type 2: {w2_type2[idx]:.4f}\n"
              f"\t\t\t Term 1: {w2_term1_type2:.4f}\n"
              f"\t\t\t Term 2: {w2_term2_type2:.4f}")

    tag = f"{dynamics.__class__.__name__} (Lipschitz={dynamics.global_lipschitz:.2f}, |C|={num_signature_points})"
    w2_bounds = {'gl': w2_global_lipschitz, 'type2': w2_type2}
    if run_type1:
        w2_bounds['type1'] = w2_type1
    return w2_bounds, tag
