import ot
import torch
import matplotlib.pyplot as plt
from copy import copy
from typing import Union, List

import discretize_distributions as ds
import GMMWas
import wasserstein
from dynamics import Dynamics
from plot import plot_multi_step


def get_initial_dist(loc_initial_dist, variance_initial_dist, **kwargs):
    return construct_diag_gaussian_dist(loc_initial_dist, variance_initial_dist)

def get_noise_dist(loc_noise_dist, variance_noise_dist, **kwargs):
    return construct_diag_gaussian_dist(loc_noise_dist, variance_noise_dist)

def construct_diag_gaussian_dist(loc_dist: Union[list, torch.Tensor], variance_dist: Union[list, torch.Tensor]):
    loc_dist = torch.as_tensor(loc_dist)
    covariance_dist = torch.diag(torch.as_tensor(variance_dist))
    return ds.MultivariateNormal(loc=loc_dist, covariance_matrix=covariance_dist)


def multi_step(dynamics: Dynamics,
               loc_noise_dist: torch.Tensor,
               variance_noise_dist: torch.Tensor,
               loc_initial_dist: torch.Tensor,
               variance_initial_dist: torch.Tensor,
               num_samples: int,
               num_time_steps: int,
               num_locs: int,
               lr: float = 0.01,
               num_iterations: int = 100,
               plot: bool = False,
               **kwargs):

    # Get initial / noise distributions parameters
    loc_noise_dist = torch.tensor(loc_noise_dist)
    covariance_noise_dist = torch.diag(torch.tensor(variance_noise_dist))
    loc_initial_dist = torch.tensor(loc_initial_dist)
    covariance_initial_dist = torch.diag(torch.tensor(variance_initial_dist))

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

    # Store wasserstein error bounds
    w2_empirical = torch.zeros(num_time_steps+1)
    w2_global_lipschitz = torch.zeros(num_time_steps+1)
    w2_independent_coupling = torch.zeros(num_time_steps+1)
    w2_lagrangian_duality = torch.zeros(num_time_steps + 1)

    # Store sample trajectories
    p_trajectories = q0_samples.clone()
    q_trajectories = q0_samples.clone()

    # Propagate the system
    for k in range(num_time_steps):
        ### Generate the signature of the approximate state distribution at time k:
        sign_q0 = ds.discretization_generator(dist=q0, num_locs=num_locs)

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
        q1_compr.compress(n_max=num_locs)
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
        w2_lagrangian_duality[k + 1] = wasserstein.compute_w2_f_p__f_disc_q_lagrangian_duality(
            sign_q0, dynamics, w2_q__disc_q=sign_q0.w2 + w2_compr, w2_p__q=w2_lagrangian_duality[k], lr=lr,
            num_iterations=num_iterations)

        ### Independent coupling method
        w2_independent_coupling[k + 1] = wasserstein.compute_w2_f_p__f_disc_q_independent_coupling(
            sign_q0, dynamics, w2_q__disc_q=sign_q0.w2 + w2_compr, w2_p__q=w2_independent_coupling[k], lr=lr,
            num_iterations=num_iterations)

        print(f"Bounds on W_2(W_2(p_{k+1}, q_{k+1})) via:\n"
              f"\t Global Lipschitz: {w2_global_lipschitz[k+1]:.4f}\n"
              f"\t Lagrangian duality: {w2_lagrangian_duality[k+1]:.4f}\n"
              f"\t Independent coupling: {w2_independent_coupling[k+1]:.4f}\n"
              f"\t Empirical: {w2_empirical[k+1]:.4f}\n")

        # Overwrite for next iteration
        q0 = q1_compr
        q0_samples = q1_compr_samples
        p0_samples = p1_samples

        if plot:
            q_trajectories = torch.cat((q_trajectories, q0_samples), dim=0)
            p_trajectories = torch.cat((p_trajectories, p0_samples), dim=0)

    tag = f"{dynamics.__class__.__name__} (Lipschitz={dynamics.global_lipschitz:.2f}, |C|={num_locs})"
    w2_bounds = {
        'emp': w2_empirical,
        'gl': w2_global_lipschitz,
        'lagr_dual': w2_lagrangian_duality,
        'indep_coupl': w2_independent_coupling
    }

    if plot:
        plot_multi_step(dynamics, w2_bounds, tag) #TODO: Change to plot trajectories?

    return w2_bounds, tag


def single_step(
        dynamics: Dynamics,
        noise_dist: ds.MultivariateNormal,
        q: Union[ds.MultivariateNormal, ds.MixtureMultivariateNormal],
        num_samples: int,
        num_locs: int,
        plot: bool = False,
        lr: float = 0.01,
        num_iterations: int = 100,
        w2_p__q_global_lipschitz: float = 0.,
        w2_p__q_independent_coupling: float = 0.,
        w2_p__q_lagrangian_duality: float = 0.,
        run_independent_coupling: bool = True,
        run_lagrangian_duality: bool = True,
        run_empirical: bool = False,
        p_samples: torch.Tensor = None,
        **kwargs):

    # Initialize System Dynamics
    print(f"Global Lipschitz constant of f: {dynamics.global_lipschitz}")

    if dynamics.num_dims == 1 and plot:
        # Plot dynamics
        x = torch.linspace(start=-5, end=5, steps=500).view(-1, 1)
        y = dynamics(x)

        fig_dynamics = plt.figure()
        plt.plot(x, y)
        plt.title("Dynamics f(x)")
        plt.show()

    # Initialize state distributions
    q_samples = q.sample(torch.Size((num_samples,)))

    # Propagate the system
    sign_q = ds.discretization_generator(dist=q, num_locs=num_locs)

    ### Propagate the (approximate) state distribution over the dynamics
    q1 = ds.MixtureMultivariateNormal(
        mixture_distribution=torch.distributions.Categorical(
            probs=sign_q.probs),
        component_distribution=ds.MultivariateNormal(
            loc=dynamics(sign_q.locs) + noise_dist.loc,
            covariance_matrix=noise_dist.covariance_matrix
        ))

    # Compress the mixture distribution
    with torch.no_grad():
        q1_pre_compression = copy(q1)
        # \todo make the unique(), i.e., the filtering in .compress() optional. Currently, it is always applied. This is problematic because GMMWas.w2 is an over-approximation, such that the w2 between the true and filtered are not guaranteed to be zero..
        if num_locs <= q1.num_components:
            w2_compr = 0.
        else:
            q1.compress(n_max=num_locs) # \todo create seperate varialbe n_max
            w2_compr = GMMWas.w2(q1, q1_pre_compression)

    q1_samples = q1.sample(torch.Size((num_samples,)))
    p1_samples = (dynamics(p_samples if p_samples is not None else q_samples) +
                  noise_dist.sample(torch.Size((num_samples,))))

    if dynamics.num_dims == 1 and plot:
        fig_propagation = plt.figure()
        plt.hist(q_samples.squeeze(), alpha=0.5, label='q', bins=100, density=True)
        plt.hist(p1_samples.squeeze(), alpha=0.5, label='f#q', bins=100, density=True)
        plt.legend()
        plt.title(f"Histograms of q and f#q")
        plt.show()

        fig_signature = plt.figure()
        plt.bar(sign_q.locs.squeeze(), sign_q.probs, width=0.1)
        plt.hist(q_samples, alpha=0.5, label='q', bins=100, density=True)
        plt.title(f"Signature of q and Histogram of q")
        plt.show()

    #### Compute W_2(p_1, q_1) = W_2(f#p_k, f#\Delta_C#q_k)
    w2_bounds = {'sign_q': sign_q.w2}

    if run_empirical:
        w2_bounds['empirical'] = ot.solve_sample(p1_samples.view(-1, dynamics.num_dims),
                                            q1_samples.view(-1, dynamics.num_dims)
                                            ).value.sqrt()

    w2_bounds['global_lipschitz'] = dynamics.global_lipschitz * (sign_q.w2 + w2_compr + w2_p__q_global_lipschitz)

    if run_independent_coupling:
        print(f"-- Independent Coupling --")
        w2_bounds['independent_coupling'] = wasserstein.compute_w2_f_p__f_disc_q_independent_coupling(
            sign_q, dynamics, w2_q__disc_q=sign_q.w2, w2_p__q=w2_p__q_independent_coupling + w2_compr, lr=lr, num_iterations=num_iterations)

    if run_lagrangian_duality:
        print(f"-- Lagrangian Duality --")
        w2_bounds['lagrangian_duality'] = wasserstein.compute_w2_f_p__f_disc_q_lagrangian_duality(
            sign_q, dynamics, w2_q__disc_q=sign_q.w2, w2_p__q=w2_p__q_lagrangian_duality + w2_compr, lr=lr, num_iterations=num_iterations)

    return w2_bounds, q1, {'q': q1_samples, 'p': p1_samples}


def single_step_w2_options(
        w2_p__q_options: Union[List[float], float],
        **kwargs):

    if isinstance(w2_p__q_options, float):
        w2_p__q_options = [w2_p__q_options]

    # store wasserstein error bounds
    w2_bounds = dict()

    #### Compute W_2(p_1, q_1) = W_2(f#p_k, f#\Delta_C#q_k)
    for w2_p__q in w2_p__q_options:
        print(f"\n ------ W_2(p,q) = {w2_p__q} ------ \n")

        w2_bounds[w2_p__q], q, samples = single_step(
            w2_p__q_global_lipschitz=w2_p__q,
            w2_p__q_independent_coupling=w2_p__q,
            w2_p__q_lagrangian_duality=w2_p__q,
            **kwargs
        )

        print(
            f"Bounds on W_2(W_2(f#p, f#disc#q)) for W_2(p,q) = {w2_p__q} and "
            f"W_2(q_0, Delta_C#q_0) = {w2_bounds[w2_p__q]['sign_q']:.4f} via:\n"
            f"\t Global Lipschits: {w2_bounds[w2_p__q]['global_lipschitz']:.4f}\n")
        print(f"\t Empirical: {w2_bounds[w2_p__q]['empirical']:.4f}\n"
              if 'empirical' in w2_bounds[w2_p__q] else "")
        print(f"\t Independent Coupling: {w2_bounds[w2_p__q]['independent_coupling']:.4f}\n"
              if 'independent_coupling' in w2_bounds[w2_p__q] else "")
        print(f"\t Lagrangian Duality: {w2_bounds[w2_p__q]['lagrangian_duality']:.4f}\n"
              if 'lagrangian_duality' in w2_bounds[w2_p__q] else "")

    return w2_bounds

        if run_independent_coupling:
            print(f"-- Independent Coupling --")
            w2_independent_coupling[idx] = wasserstein.compute_w2_f_p__f_disc_q_independent_coupling(
                sign_q, dynamics, w2_q__disc_q=sign_q.w2, w2_p__q=w2_p__q + w2_compr, lr=lr, num_iterations=num_iterations)

        if run_lagrangian_duality:
            print(f"-- Lagrangian Duality --")
            w2_lagrangian_duality[idx] = wasserstein.compute_w2_f_p__f_disc_q_lagrangian_duality(
                sign_q, dynamics, w2_q__disc_q=sign_q.w2, w2_p__q=w2_p__q + w2_compr, lr=lr, num_iterations=num_iterations)

        print(f"Bounds on W_2(W_2(f#p, f#disc#q)) for W_2(p,q) = {w2_p__q} and W_2(q_0, Delta_C#q_0) = {sign_q.w2:.4f} via:\n"
              f"\t Global Lipschits: {w2_global_lipschitz[idx]:.4f}\n")
        if run_empirical:
            print(f"\t Empirical: {w2_empirical[idx]:.4f}\n")
        if run_independent_coupling:
            print(f"\t Independent Coupling: {w2_independent_coupling[idx]:.4f}\n")
        if run_lagrangian_duality:
            print(f"\t Lagrangian Duality: {w2_lagrangian_duality[idx]:.4f}\n")

    w2_bounds = {'gl': w2_global_lipschitz}
    if run_empirical:
        w2_bounds['empirical'] = w2_empirical
    if run_independent_coupling:
        w2_bounds['independent_coupling'] = w2_independent_coupling
    if run_lagrangian_duality:
        w2_bounds['lagrangian_duality'] = w2_lagrangian_duality

    # loop over time steps
    for k in range(num_time_steps):
        print(f'---- TIME STEP {k} ----')
        w2_bounds, q, samples = single_step(
            dynamics=dynamics,
            noise_dist=noise_dist,
            q=q,
            p_samples=p_trajectories[k] if k==0 else None,
            num_samples=num_samples,
            w2_p__q_options=w2_empirical[k] if k > 0 else 0.,
            **kwargs
        )

        w2_empirical[k+1] = w2_bounds['empirical']
        w2_global_lipschitz[k + 1] = w2_bounds['gl']
        w2_independent_coupling[k+1] = w2_bounds['independent_coupling']
        w2_lagrangian_duality[k+1] = w2_bounds['lagrangian_duality']

        p_trajectories[k] = samples['p']
        q_trajectories[k] = samples['q']

    w2_bounds = {
        'empirical': w2_empirical,
        'gl': w2_global_lipschitz,
        'lagr_dual': w2_lagrangian_duality,
        'indep_coupl': w2_independent_coupling
    }

    return w2_bounds, p_trajectories, q_trajectories
