import ot
import torch
from copy import copy
from typing import Union, List, Optional

import discretize_distributions as ds
import GMMWas
import wasserstein
from dynamics import Dynamics, AdditiveGaussianDynamics
from plot import plot_multi_step
from utils_distributions import cross_product, sum_discrete_distributions, quantize

def _propagate_via_gmms(
        dynamics: Dynamics,
        noise_dist: ds.MultivariateNormal,
        sign_state_dist: Union[ds.DiscretizedMultivariateNormal, ds.CategoricalFloat]
):
    if isinstance(dynamics, AdditiveGaussianDynamics):
        assert isinstance(noise_dist, ds.MultivariateNormal)
        sign_q = sign_state_dist  # \todo make diff between sign_Q and signature of noise and state more clear
        q1 = ds.MixtureMultivariateNormal(
            mixture_distribution=torch.distributions.Categorical(
                probs=sign_state_dist.probs),
            component_distribution=ds.MultivariateNormal(
                loc=dynamics.state_dynamics(sign_state_dist.locs) + noise_dist.loc,
                covariance_matrix=noise_dist.covariance_matrix
            ))
        return sign_q, q1
    else:
        raise ValueError('Propagation via GMM not possible for non additive Gaussian noise.')


def _propagate_state_noise_discrete_additive(
        dynamics: Dynamics,
        noise_dist: Union[ds.DiscretizedMultivariateNormal, ds.CategoricalFloat],
        sign_state_dist: Union[ds.DiscretizedMultivariateNormal, ds.CategoricalFloat]
):
    sign_q = sign_state_dist
    propagated_states = ds.CategoricalFloat(probs=sign_state_dist.probs,
                                            locs=dynamics.state_dynamics(sign_state_dist.locs))
    sum_probs, sum_locs = sum_discrete_distributions(propagated_states, noise_dist)
    q1 = ds.CategoricalFloat(probs=sum_probs, locs=sum_locs)

    return sign_q, q1

def _propagate_state_noise_discrete_general(
        dynamics: Dynamics,
        noise_dist: Union[ds.DiscretizedMultivariateNormal, ds.CategoricalFloat],
        sign_state_dist: Union[ds.DiscretizedMultivariateNormal, ds.CategoricalFloat]
):
    cross_probs, cross_locs = cross_product(sign_state_dist, noise_dist)
    sign_q = ds.CategoricalFloat(probs=cross_probs, locs=cross_locs)
    q1 = ds.CategoricalFloat(probs=cross_probs, locs=dynamics(cross_locs))

    return sign_q, q1

def propagate_state_dist_over_dynamics(
        dynamics: Dynamics,
        noise_dist: Union[ds.MultivariateNormal, ds.DiscretizedMultivariateNormal, ds.CategoricalFloat],
        sign_state_dist: Union[ds.DiscretizedMultivariateNormal, ds.CategoricalFloat],
        propagate_via_gmm: bool = False
):
    if propagate_via_gmm:
        sign_q, q1 = _propagate_via_gmms(dynamics, noise_dist, sign_state_dist)
    else:
        if isinstance(dynamics, AdditiveGaussianDynamics):
            sign_q, q1 = _propagate_state_noise_discrete_additive(dynamics, noise_dist, sign_state_dist)

        else:
            sign_q, q1 = _propagate_state_noise_discrete_general(dynamics, noise_dist, sign_state_dist)

    return sign_q, q1

def single_step(
        dynamics: Dynamics,
        noise_dist: ds.MultivariateNormal,
        q: Union[ds.MultivariateNormal, ds.MixtureMultivariateNormal],
        num_samples: int,
        num_locs: int,
        w2_p__q_global_lipschitz: float = 0.,
        w2_p__q_lagrangian_duality: float = 0.,
        run_lagrangian_duality: bool = True,
        run_empirical: bool = False,
        p_samples: Optional[torch.Tensor] = None,
        num_locs_after_compr: Optional[int] = None,
        propagate_via_gmm: bool = False,
        **kwargs):

    # Initialize System Dynamics
    print(f"Global Lipschitz constant of f: {dynamics.global_lipschitz}")

    # Compress the mixture distribution
    with torch.no_grad():
        # \todo make the unique(), i.e., the filtering in .compress() optional. Currently, it is always applied. This is problematic because GMMWas.w2 is an over-approximation, such that the w2 between the true and filtered are not guaranteed to be zero..
        if isinstance(q, ds.MultivariateNormal) or (num_locs if num_locs_after_compr is None else num_locs_after_compr) >= q.num_components:
            w2_compr = 0.
        else:
            q_pre_compression = copy(q)
            q.compress(n_max=num_locs if num_locs_after_compr is None else num_locs_after_compr)
            w2_compr = GMMWas.w2(q, q_pre_compression)

    # Approximate the state distribution
    sign_q, theta_d = quantize(q, num_locs)

    # Approximate the noise distribution
    if not propagate_via_gmm:
        sign_noise_dist, w2_noise_quantization = quantize(noise_dist, num_locs)
        sign_q, q1 = propagate_state_dist_over_dynamics(dynamics, sign_noise_dist, sign_q)
    else:
        # Propagate the (approximate) state distribution over the dynamics
        sign_q, q1 = propagate_state_dist_over_dynamics(dynamics, noise_dist, sign_q)

    # Empirically approximate the state distribution
    q_samples = q.sample(torch.Size((num_samples,)))
    q1_samples = q1.sample(torch.Size((num_samples,)))
    noise_samples = noise_dist.sample(torch.Size((num_samples,)))

    p1_samples = dynamics(torch.cat((p_samples if p_samples is not None else q_samples, noise_samples), dim=-1))

    #### Compute W_2(p_1, q_1) = W_2(f#p_k, f#\Delta_C#q_k)
    w2_bounds = {'sign_q': theta_d,
                 'empirical': torch.nan,
                 'lagrangian_duality': torch.nan
                 }

    if run_empirical:
        w2_bounds['empirical'] = ot.solve_sample(p1_samples.view(-1, p1_samples.shape[-1]),
                                            q1_samples.view(-1, q1_samples.shape[-1])
                                            ).value.sqrt()

    if isinstance(dynamics, AdditiveGaussianDynamics):
        f = dynamics.state_dynamics
    else:
        f = dynamics
        theta_d += w2_noise_quantization

    w2_bounds['global_lipschitz'] = dynamics.global_lipschitz * (theta_d + w2_compr + w2_p__q_global_lipschitz)

    if run_lagrangian_duality:
        print(f"-- Lagrangian Duality --")
        w2_bounds['lagrangian_duality'] = wasserstein.compute_w2_f_p__f_disc_q_lagrangian_duality(
            signature=sign_q, f=f, w2_q__disc_q=theta_d, w2_p__q=w2_p__q_lagrangian_duality + w2_compr, **kwargs)

    if isinstance(dynamics, AdditiveGaussianDynamics) and not propagate_via_gmm:
        w2_bounds['global_lipschitz'] += w2_noise_quantization
        w2_bounds['lagrangian_duality'] += w2_noise_quantization

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
            w2_p__q_lagrangian_duality=w2_p__q,
            **kwargs
        )

        print(
            f"Bounds on W_2(f#p, f#disc#q) for W_2(p,q) = {w2_p__q} and "
            f"W_2(q_0, Delta_C#q_0) = {w2_bounds[w2_p__q]['sign_q']:.4f} via:\n"
            f"\t Global Lipschitz: {w2_bounds[w2_p__q]['global_lipschitz']:.4f}\n")
        print(f"\t Empirical: {w2_bounds[w2_p__q]['empirical']:.4f}\n"
              if 'empirical' in w2_bounds[w2_p__q] else "")
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
    w2_bounds = {0: {'global_lipschitz': 0., 'lagrangian_duality': 0.}}

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
            w2_p__q_lagrangian_duality=w2_bounds[k]['lagrangian_duality'],
            **kwargs
        )

        print(
            f"Bounds on W_2(p_{k+1}, q_{k+1}) via:\n"
            f"\t Global Lipschitz: {w2_bounds[k+1]['global_lipschitz']:.4f}\n")
        print(f"\t Empirical: {w2_bounds[k+1]['empirical']:.4f}\n"
              if 'empirical' in w2_bounds[k+1] else "")
        print(f"\t Lagrangian Duality: {w2_bounds[k+1]['lagrangian_duality']:.4f}\n"
              if 'lagrangian_duality' in w2_bounds[k+1] else "")

    return w2_bounds, samples
