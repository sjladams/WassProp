import ot
import torch
import matplotlib.pyplot as plt
from copy import copy
from typing import Union, List, Optional

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
        p_samples: Optional[torch.Tensor] = None,
        num_locs_after_compr: Optional[int] = None,
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

    # Compress the mixture distribution
    with torch.no_grad():
        q_pre_compression = copy(q)
        # \todo make the unique(), i.e., the filtering in .compress() optional. Currently, it is always applied. This is problematic because GMMWas.w2 is an over-approximation, such that the w2 between the true and filtered are not guaranteed to be zero..
        if isinstance(q, ds.MultivariateNormal) or num_locs >= q.num_components:
            w2_compr = 0.
        else:
            q.compress(n_max=num_locs)  # \todo create seperate varialbe n_max
            w2_compr = GMMWas.w2(q, q_pre_compression)

    # Approximate the state distribution
    sign_q = ds.discretization_generator(dist=q, num_locs=num_locs)

    # Propagate the (approximate) state distribution over the dynamics
    q1 = ds.MixtureMultivariateNormal(
        mixture_distribution=torch.distributions.Categorical(
            probs=sign_q.probs),
        component_distribution=ds.MultivariateNormal(
            loc=dynamics(sign_q.locs) + noise_dist.loc,
            covariance_matrix=noise_dist.covariance_matrix
        ))

    # Empirically approximate the state distribution
    q_samples = q.sample(torch.Size((num_samples,)))
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


def multi_step(
        dynamics: Dynamics,
        noise_dist: ds.MultivariateNormal,
        q: Union[ds.MultivariateNormal, ds.MixtureMultivariateNormal],
        num_time_steps: int,
        **kwargs):

    # Initialize w2_p__q error:
    w2_bounds = {0: {'global_lipschitz': 0., 'independent_coupling': 0., 'lagrangian_duality': 0.}}

    # store trajectories
    samples = dict()

    # loop over time steps
    for k in range(num_time_steps):
        print(f'---- TIME STEP {k} ----')
        w2_bounds[k+1], q, samples[k] = single_step(
            dynamics=dynamics,
            noise_dist=noise_dist,
            q=q,
            p_samples=None if k==0 else samples[k-1]['p'],
            w2_p__q_global_lipschitz=w2_bounds[k]['global_lipschitz'],
            w2_p__q_independent_coupling=w2_bounds[k]['independent_coupling'],
            w2_p__q_lagrangian_duality=w2_bounds[k]['lagrangian_duality'],
            **kwargs
        )

        print(
            f"Bounds on W_2(W_2(p_{k+1}, q_{k+1})) via:\n"
            f"\t Global Lipschits: {w2_bounds[k+1]['global_lipschitz']:.4f}\n")
        print(f"\t Empirical: {w2_bounds[k+1]['empirical']:.4f}\n"
              if 'empirical' in w2_bounds[k+1] else "")
        print(f"\t Independent Coupling: {w2_bounds[k+1]['independent_coupling']:.4f}\n"
              if 'independent_coupling' in w2_bounds[k+1] else "")
        print(f"\t Lagrangian Duality: {w2_bounds[k+1]['lagrangian_duality']:.4f}\n"
              if 'lagrangian_duality' in w2_bounds[k+1] else "")

    return w2_bounds, samples
