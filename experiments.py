import ot
import torch
from copy import copy
from typing import Union, List, Optional

import discretize_distributions as ds
import GMMWas
import wasserstein
from dynamics import Dynamics, AdditiveGaussianDynamics
from plot import plot_multi_step


def get_initial_dist(loc_initial_dist, variance_initial_dist, **kwargs):
    return construct_diag_gaussian_dist(loc_initial_dist, variance_initial_dist)


def get_noise_dist(loc_noise_dist, variance_noise_dist, **kwargs):
    return construct_diag_gaussian_dist(loc_noise_dist, variance_noise_dist)


def construct_diag_gaussian_dist(loc_dist: Union[list, torch.Tensor], variance_dist: Union[list, torch.Tensor]):
    loc_dist = torch.as_tensor(loc_dist)
    covariance_dist = torch.diag(torch.as_tensor(variance_dist))
    return ds.MultivariateNormal(loc=loc_dist, covariance_matrix=covariance_dist)


def propagate_state_dist_over_dynamics(
        dynamics: Dynamics,
        noise_dist: Union[ds.MultivariateNormal, ds.DiscretizedMultivariateNormal],
        sign_state_dist: Union[ds.DiscretizedMultivariateNormal, ds.CategoricalFloat]
):
    if isinstance(dynamics, AdditiveGaussianDynamics): # \todo add check on noise
        assert isinstance(noise_dist, ds.MultivariateNormal)
        sign_q = sign_state_dist # \todo make diff between sign_Q and signature of noise and state more clear
        q1 = ds.MixtureMultivariateNormal(
                mixture_distribution=torch.distributions.Categorical(
                    probs=sign_state_dist.probs),
                component_distribution=ds.MultivariateNormal(
                    loc=dynamics.state_dynamics(sign_state_dist.locs) + noise_dist.loc,
                    covariance_matrix=noise_dist.covariance_matrix
                ))
    else:
        assert isinstance(noise_dist, ds.DiscretizedMultivariateNormal)
        n, m = sign_state_dist.locs.size(0), noise_dist.locs.size(0)
        d = sign_state_dist.locs.shape[-1]
        locs_state_expanded = sign_state_dist.locs.unsqueeze(1)
        locs_noise_expanded = noise_dist.locs.unsqueeze(0)
        combinations = torch.cat((locs_state_expanded.expand(-1, m, -1), locs_noise_expanded.expand(n, -1, -1)), dim=-1)
        combinations_flat = combinations.view(-1, 2 * d)
        probs_combined = sign_state_dist.probs.unsqueeze(1) * noise_dist.probs.unsqueeze(0)
        probs_combined_flat = probs_combined.view(-1)

        sign_q = ds.CategoricalFloat(probs=probs_combined_flat, locs=combinations_flat)
        # sign q is the cross-product of the signature of the states and the noise, hence the approximation error of
        # sign_q is the sum of the errors of the two signatures:
        sign_q.w2 = noise_dist.w2 + sign_state_dist.w2 if isinstance(sign_state_dist, ds.DiscretizedMultivariateNormal) else 0.
        q1 = ds.CategoricalFloat(probs=probs_combined_flat, locs=dynamics(combinations_flat))

    return sign_q, q1


def single_step(
        dynamics: Dynamics,
        noise_dist: ds.MultivariateNormal,
        q: Union[ds.MultivariateNormal, ds.MixtureMultivariateNormal],
        num_samples: int,
        num_locs: int,
        plot: bool = False,
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

    # Compress the mixture distribution
    with torch.no_grad():
        q_pre_compression = copy(q)
        # \todo make the unique(), i.e., the filtering in .compress() optional. Currently, it is always applied. This is problematic because GMMWas.w2 is an over-approximation, such that the w2 between the true and filtered are not guaranteed to be zero..
        if isinstance(q, ds.MultivariateNormal) or (num_locs if num_locs_after_compr is None else num_locs_after_compr) >= q.num_components:
            w2_compr = 0.
        else:
            q.compress(n_max=num_locs if num_locs_after_compr is None else num_locs_after_compr)
            w2_compr = GMMWas.w2(q, q_pre_compression)

    # Approximate the state distribution
    sign_q = ds.discretization_generator(dist=q, num_locs=num_locs)

    # Approximate the noise distribution
    if not isinstance(dynamics, AdditiveGaussianDynamics):
        noise_dist = ds.discretization_generator(dist=noise_dist, num_locs=num_locs)

    # Propagate the (approximate) state distribution over the dynamics
    sign_q, q1 = propagate_state_dist_over_dynamics(dynamics, noise_dist, sign_q)

    # Empirically approximate the state distribution
    q_samples = q.sample(torch.Size((num_samples,)))
    q1_samples = q1.sample(torch.Size((num_samples,)))
    noise_samples = noise_dist.sample(torch.Size((num_samples,)))

    p1_samples = dynamics(torch.cat((p_samples if p_samples is not None else q_samples, noise_samples), dim=-1))

    #### Compute W_2(p_1, q_1) = W_2(f#p_k, f#\Delta_C#q_k)
    w2_bounds = {'sign_q': sign_q.w2,
                 'empirical': torch.nan,
                 'independent_coupling': torch.nan,
                 'lagrangian_duality': torch.nan
                 }

    if run_empirical:
        w2_bounds['empirical'] = ot.solve_sample(p1_samples.view(-1, dynamics.num_dims),
                                            q1_samples.view(-1, dynamics.num_dims)
                                            ).value.sqrt()

    w2_bounds['global_lipschitz'] = dynamics.global_lipschitz * (sign_q.w2 + w2_compr + w2_p__q_global_lipschitz)

    if isinstance(dynamics, AdditiveGaussianDynamics):
        f = dynamics.state_dynamics
    else:
        f = dynamics

    if run_independent_coupling:
        print(f"-- Independent Coupling --")
        w2_bounds['independent_coupling'] = wasserstein.compute_w2_f_p__f_disc_q_independent_coupling(
            signature=sign_q, f=f, w2_q__disc_q=sign_q.w2, w2_p__q=w2_p__q_independent_coupling + w2_compr, **kwargs)

    if run_lagrangian_duality:
        print(f"-- Lagrangian Duality --")
        w2_bounds['lagrangian_duality'] = wasserstein.compute_w2_f_p__f_disc_q_lagrangian_duality(
            signature=sign_q, f=f, w2_q__disc_q=sign_q.w2, w2_p__q=w2_p__q_lagrangian_duality + w2_compr, **kwargs)

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
            f"Bounds on W_2(f#p, f#disc#q) for W_2(p,q) = {w2_p__q} and "
            f"W_2(q_0, Delta_C#q_0) = {w2_bounds[w2_p__q]['sign_q']:.4f} via:\n"
            f"\t Global Lipschitz: {w2_bounds[w2_p__q]['global_lipschitz']:.4f}\n")
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
        optimize_locs: bool = False,
        **kwargs):

    if optimize_locs:
        raise NotImplementedError("Optimization of the signature locations is not yet implemented for multi_step.")

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
            f"Bounds on W_2(p_{k+1}, q_{k+1}) via:\n"
            f"\t Global Lipschitz: {w2_bounds[k+1]['global_lipschitz']:.4f}\n")
        print(f"\t Empirical: {w2_bounds[k+1]['empirical']:.4f}\n"
              if 'empirical' in w2_bounds[k+1] else "")
        print(f"\t Independent Coupling: {w2_bounds[k+1]['independent_coupling']:.4f}\n"
              if 'independent_coupling' in w2_bounds[k+1] else "")
        print(f"\t Lagrangian Duality: {w2_bounds[k+1]['lagrangian_duality']:.4f}\n"
              if 'lagrangian_duality' in w2_bounds[k+1] else "")

    return w2_bounds, samples
