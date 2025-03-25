import ot
import torch
from copy import copy
from typing import Union, List, Optional

import discretize_distributions as ds
import GMMWas
import wasserstein
from dynamics import Dynamics, AdditiveGaussianDynamics
import propagation as prop


def single_step(
        dynamics: Dynamics,
        noise_dist: ds.MultivariateNormal,
        q: Union[ds.MultivariateNormal, ds.MixtureMultivariateNormal, ds.CategoricalFloat],
        num_samples: int,
        num_locs: int,
        propagate_via_gmm: bool, # todo rename
        w2_p__q_global_lipschitz: float = 0.,
        w2_p__q_lagrangian_duality: float = 0.,
        run_lagrangian_duality: bool = True,
        run_empirical: bool = False,
        p_samples: Optional[torch.Tensor] = None,
        num_locs_after_compr: Optional[int] = None):

    if num_locs_after_compr is None:
        num_locs_after_compr = num_locs

    # Initialize System Dynamics
    print(f"Global Lipschitz constant of f: {dynamics.global_lipschitz}")

    # Compress the mixture distribution
    with torch.no_grad():
        if isinstance(q, ds.MultivariateNormal) or num_locs_after_compr >= q.num_components:
            w2_compr = 0.
        else:
            if isinstance(q, ds.MixtureMultivariateNormal):
                q = ds.unique_mixture_multivariate_normal(q)
                if num_locs_after_compr >= q.num_components:
                    w2_compr = 0.
                else:
                    q_pre = copy(q)
                    q = ds.compress_mixture_multivariate_normal(q, n_max=num_locs_after_compr)
                    w2_compr = GMMWas.w2(q, q_pre)
            elif isinstance(q, ds.CategoricalFloat):
                q_pre = copy(q)
                q = ds.compress_categorical_floats(q_pre, n_max=num_locs_after_compr)
                w2_compr = GMMWas.w2(q, q_pre)
            else:
                raise ValueError

    # Approximate the state distribution
    sign_q = ds.discretization_generator(q, num_locs)

    # Propagate
    if isinstance(dynamics, AdditiveGaussianDynamics) and propagate_via_gmm:
        q1 = prop.propagate_additive_gaussian_noise(dynamics, noise_dist, sign_q)
    else:
        sign_noise_dist = ds.discretization_generator(noise_dist, num_locs)
        if isinstance(dynamics, AdditiveGaussianDynamics):
            q1 = prop.propagate_additive_discrete_noise(dynamics, sign_noise_dist, sign_q)
        else:
            sign_cross, q1 = prop.propagate_general_discrete_noise(dynamics, sign_noise_dist, sign_q)

    # Empirically approximate the state distribution
    q_samples = q.sample(torch.Size((num_samples,)))
    p_samples = p_samples if p_samples is not None else q_samples
    q1_samples = q1.sample(torch.Size((num_samples,)))
    noise_samples = noise_dist.sample(torch.Size((num_samples,)))
    p1_samples = dynamics(torch.cat((p_samples, noise_samples), dim=-1))

    #### Compute W_2(p_1, q_1) = W_2(f#p_k, f#\Delta_C#q_k)
    if run_empirical:
        w2_p1__q1_empirical = ot.solve_sample(p1_samples.view(-1, p1_samples.shape[-1]),
                                                 q1_samples.view(-1, q1_samples.shape[-1])
                                                 ).value.sqrt()
    else:
        w2_p1__q1_empirical = torch.nan

    w2_p1__q1_global_lipschitz = dynamics.global_lipschitz * (sign_q.w2 + w2_compr + w2_p__q_global_lipschitz)
    if isinstance(dynamics, AdditiveGaussianDynamics) and not propagate_via_gmm:
        w2_p1__q1_global_lipschitz += sign_noise_dist.w2
    elif not isinstance(dynamics, AdditiveGaussianDynamics):
        w2_p1__q1_global_lipschitz += dynamics.global_lipschitz * sign_noise_dist.w2

    if run_lagrangian_duality:
        print(f"-- Lagrangian Duality --")
        if isinstance(dynamics, AdditiveGaussianDynamics):
            w2_p1__q1_lagrangian_duality = wasserstein.compute_w2_f_p__f_disc_q_lagrangian_duality(
                signature=sign_q, f=dynamics.state_dynamics, w2_q__disc_q=sign_q.w2, w2_p__q=w2_p__q_lagrangian_duality + w2_compr)
            if not propagate_via_gmm:
                w2_p1__q1_lagrangian_duality += sign_noise_dist.w2
        else:
            w2_p1__q1_lagrangian_duality = wasserstein.compute_w2_f_p__f_disc_q_lagrangian_duality(
                signature=sign_cross, f=dynamics, w2_q__disc_q=sign_q.w2+sign_noise_dist.w2, w2_p__q=w2_p__q_lagrangian_duality + w2_compr)
    else:
        w2_p1__q1_lagrangian_duality = torch.nan

    return dict(
        w2_q__sign_q=sign_q.w2,
        w2_p1__q1_empirical=w2_p1__q1_empirical,
        w2_p1__q1_global_lipschitz=w2_p1__q1_global_lipschitz,
        w2_p1__q1_lagrangian_duality=w2_p1__q1_lagrangian_duality,
        q1=q1,
        q1_samples=q1_samples,
        p1_samples=p1_samples
    )


def single_step_w2_options(
        w2_p__q_options: Union[List[float], float],
        **kwargs):

    if isinstance(w2_p__q_options, float):
        w2_p__q_options = [w2_p__q_options]

    # store wasserstein error bounds
    w2_p1__q1_store = dict()
    w2_q__sign_q_store = dict()

    #### Compute W_2(p_1, q_1) = W_2(f#p_k, f#\Delta_C#q_k)
    for w2_p__q in w2_p__q_options:
        print(f"\n ------ W_2(p,q) = {w2_p__q} ------ \n")

        out = single_step(
            w2_p__q_global_lipschitz=w2_p__q,
            w2_p__q_lagrangian_duality=w2_p__q,
            **kwargs
        )
        w2_p1__q1_store[w2_p__q] = {key: value for key, value in out.items() if 'w2_p1__q1' in key}
        w2_q__sign_q_store[w2_p__q] = out['w2_q__sign_q']

        print(
            f"Bounds on W_2(f#p, f#disc#q) for W_2(p,q) = {w2_p__q} and "
            f"W_2(q_0, Delta_C#q_0) = {out['w2_q__sign_q']:.4f} via:\n"
            f"\t Global Lipschitz: {out['w2_p1__q1_global_lipschitz']:.4f}\n"
            f"\t Empirical: {out['w2_p1__q1_empirical']:.4f}\n"
            f"\t Lagrangian Duality: {out['w2_p1__q1_lagrangian_duality']:.4f}\n")

    return w2_q__sign_q_store, w2_p1__q1_store


def multi_step( # \todo kill gradients
        dynamics: Dynamics,
        noise_dist: ds.MultivariateNormal,
        q: Union[ds.MultivariateNormal, ds.MixtureMultivariateNormal],
        num_time_steps: int,
        **kwargs):

    # store wasserstein error bounds
    w2_p1__q1_store = {0: dict(w2_p1__q1_global_lipschitz=0., w2_p1__q1_lagrangian_duality=0.)}
    w2_q__sign_q_store = dict()

    # store trajectories
    samples_store = dict()

    # loop over time steps
    for k in range(num_time_steps):
        print(f'---- TIME STEP {k} ----')
        out = single_step(
            dynamics=dynamics,
            noise_dist=noise_dist,
            q=q,
            p_samples=samples_store[k-1]['p1_samples'] if k-1 in samples_store else None,
            w2_p__q_global_lipschitz=w2_p1__q1_store[k]['w2_p1__q1_global_lipschitz'],
            w2_p__q_lagrangian_duality=w2_p1__q1_store[k]['w2_p1__q1_lagrangian_duality'],
            **kwargs
        )
        q = out['q1']
        w2_p1__q1_store[k+1] = {key: value for key, value in out.items() if 'w2_p1__q1' in key}
        w2_q__sign_q_store[k+1] = out['w2_q__sign_q']
        samples_store[k] = {key: value for key, value in out.items() if 'samples' in key}

        print(
            f"Bounds on W_2(p_{k+1}, q_{k+1}) via:\n"
            f"\t Global Lipschitz: {out['w2_p1__q1_global_lipschitz']:.4f}\n"
            f"\t Empirical: {out['w2_p1__q1_empirical']:.4f}\n"
            f"\t Lagrangian Duality: {out['w2_p1__q1_lagrangian_duality']:.4f}\n"
        )

    return w2_q__sign_q_store, w2_p1__q1_store, samples_store
