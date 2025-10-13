import ot
import torch
from typing import Union, List, Optional

import discretize_distributions as dd
import discretize_distributions.distributions as dd_dists
import wasserstein
from dynamics import Dynamics, AdditiveGaussianDynamics
import propagation as prop
from utils_distributions import compress_compress_categorical_floats, sample_from_ambiguity_set


def single_step(dynamics: Dynamics, **kwargs):
    if isinstance(dynamics, AdditiveGaussianDynamics):
        return single_step_additive_dynamics(dynamics=dynamics, **kwargs)
    else:
        return single_step_general_dynamics(dynamics=dynamics, **kwargs)

def single_step_general_dynamics(
        dynamics: Dynamics,
        noise_dist: Union[dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal],
        q: Union[dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal, dd_dists.CategoricalFloat],
        num_samples: int,
        num_locs: int,
        w2_noise_dist: float = 0.,
        w2_p__q_global_lipschitz: float = 0.,
        w2_p__q_lagrangian_duality: float = 0.,
        run_lagrangian_duality: bool = True,
        run_empirical: bool = False,
        p_samples: Optional[torch.Tensor] = None,
    ):

    # Approximate the state distribution
    if isinstance(q, dd_dists.CategoricalFloat):
        disc_q, w2_q__disc_q = compress_compress_categorical_floats(q, size_after_compr=num_locs)
    else:
        scheme_q = dd.generate_scheme(dist=q, scheme_size=num_locs)
        disc_q, w2_q__disc_q = dd.discretize(q, scheme_q)

    # Approximate the noise distribution
    scheme_noise_dist = dd.generate_scheme(dist=noise_dist, scheme_size=num_locs, per_mode=False)
    disc_noise_dist, w2_noise_dist__disc_noise_dist = dd.discretize(noise_dist, scheme_noise_dist)

    # Propagate
    q1 = prop.propagate_general_discrete_noise(dynamics, disc_noise_dist, disc_q)

    # Empirically approximate the state distribution
    q_samples = q.sample(torch.Size((num_samples,)))
    p_samples = p_samples if p_samples is not None else q_samples
    q1_samples = q1.sample(torch.Size((num_samples,)))
    noise_samples = sample_from_ambiguity_set(noise_dist, w2_noise_dist, num_samples)
    p1_samples = dynamics(torch.cat((p_samples, noise_samples), dim=-1))

    #### Compute W_2(p_1, q_1) = W_2(f#p_k, f#\Delta_C#q_k)
    if run_empirical:
        w2_p1__q1_empirical = ot.solve_sample(
            p1_samples.view(-1, p1_samples.shape[-1]),
            q1_samples.view(-1, q1_samples.shape[-1])
        ).value.sqrt()
    else:
        w2_p1__q1_empirical = torch.nan

    w2_p1__q1_global_lipschitz = dynamics.global_lipschitz * (w2_q__disc_q + w2_p__q_global_lipschitz)
    w2_p1__q1_global_lipschitz += dynamics.global_lipschitz * (w2_noise_dist__disc_noise_dist + w2_noise_dist)

    if run_lagrangian_duality:
        print(f"-- Lagrangian Duality --")
        w2_p1__q1_lagrangian_duality = wasserstein.compute_w2_f_p__f_disc_q_lagrangian_duality(
            disc_q=disc_q,
            f=dynamics, 
            w2_p__disc_q=w2_q__disc_q + w2_noise_dist__disc_noise_dist + w2_p__q_lagrangian_duality + w2_noise_dist
        )
    else:
        w2_p1__q1_lagrangian_duality = torch.nan

    return dict(
        w2_q__disc_q=w2_q__disc_q,
        w2_p1__q1_empirical=w2_p1__q1_empirical,
        w2_p1__q1_global_lipschitz=w2_p1__q1_global_lipschitz,
        w2_p1__q1_lagrangian_duality=w2_p1__q1_lagrangian_duality,
        q1=q1,
        q_comp=q,
        q1_samples=q1_samples,
        p1_samples=p1_samples
    )

def single_step_additive_dynamics(
        dynamics: AdditiveGaussianDynamics,
        noise_dist: dd_dists.MultivariateNormal,
        q: Union[dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal],
        num_samples: int,
        num_locs: int,
        w2_noise_dist: float = 0.,
        w2_p__q_global_lipschitz: float = 0.,
        w2_p__q_lagrangian_duality: float = 0.,
        run_lagrangian_duality: bool = True,
        run_empirical: bool = False,
        p_samples: Optional[torch.Tensor] = None,
    ):
    if not (isinstance(q, (dd_dists.MixtureMultivariateNormal, dd_dists.MultivariateNormal))):
        raise ValueError("q should be of type MixtureMultivariateNormal or MultivariateNormal")

    # Approximate the state distribution
    scheme_q = dd.generate_scheme(dist=q, scheme_size=num_locs)
    disc_q, w2_q__disc_q = dd.discretize(q, scheme_q)

    # Propagate
    q1 = prop.propagate_additive_gaussian_noise(dynamics, noise_dist, disc_q)

    # Empirically approximate the state distribution
    q_samples = q.sample(torch.Size((num_samples,)))
    p_samples = p_samples if p_samples is not None else q_samples
    q1_samples = q1.sample(torch.Size((num_samples,)))
    noise_samples = sample_from_ambiguity_set(noise_dist, w2_noise_dist, num_samples)
    p1_samples = dynamics(torch.cat((p_samples, noise_samples), dim=-1))

    #### Compute W_2(p_1, q_1) = W_2(f#p_k, f#\Delta_C#q_k)
    if run_empirical:
        w2_p1__q1_empirical = ot.solve_sample(
            p1_samples.view(-1, p1_samples.shape[-1]),
            q1_samples.view(-1, q1_samples.shape[-1])
        ).value.sqrt()
    else:
        w2_p1__q1_empirical = torch.nan

    w2_p1__q1_global_lipschitz = dynamics.global_lipschitz * (w2_q__disc_q + w2_p__q_global_lipschitz)
    w2_p1__q1_global_lipschitz += w2_noise_dist

    if run_lagrangian_duality:
        print(f"-- Lagrangian Duality --")
        w2_p1__q1_lagrangian_duality = wasserstein.compute_w2_f_p__f_disc_q_lagrangian_duality(
            disc_q=disc_q, 
            f=dynamics.state_dynamics, 
            w2_p__disc_q=w2_q__disc_q + w2_p__q_lagrangian_duality
        )
        w2_p1__q1_lagrangian_duality += w2_noise_dist
    else:
        w2_p1__q1_lagrangian_duality = torch.nan

    return dict(
        w2_q__disc_q=w2_q__disc_q,
        w2_p1__q1_empirical=w2_p1__q1_empirical,
        w2_p1__q1_global_lipschitz=w2_p1__q1_global_lipschitz,
        w2_p1__q1_lagrangian_duality=w2_p1__q1_lagrangian_duality,
        q1=q1,
        q_comp=q,
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
    w2_q__disc_q_store = dict()

    #### Compute W_2(p_1, q_1) = W_2(f#p_k, f#\Delta_C#q_k)
    for w2_p__q in w2_p__q_options:
        print(f"\n ------ W_2(p,q) = {w2_p__q} ------ \n")

        out = single_step(
            w2_p__q_global_lipschitz=w2_p__q,
            w2_p__q_lagrangian_duality=w2_p__q,
            **kwargs
        )
        w2_p1__q1_store[w2_p__q] = {key: value for key, value in out.items() if 'w2_p1__q1' in key}
        w2_q__disc_q_store[w2_p__q] = out['w2_q__disc_q']

        print(
            f"Bounds on W_2(f#p, f#disc#q) for W_2(p,q) = {w2_p__q} and "
            f"W_2(q_0, Delta_C#q_0) = {out['w2_q__disc_q']:.4f} via:\n"
            f"\t Global Lipschitz: {out['w2_p1__q1_global_lipschitz']:.4f}\n"
            f"\t Empirical: {out['w2_p1__q1_empirical']:.4f}\n"
            f"\t Lagrangian Duality: {out['w2_p1__q1_lagrangian_duality']:.4f}\n")

    return w2_q__disc_q_store, w2_p1__q1_store


# @torch.no_grad()
def multi_step(
        dynamics: Dynamics,
        noise_dist: dd_dists.MultivariateNormal,
        q: Union[dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal],
        num_time_steps: int,
        num_samples: int,
        w2_p__q: float = 0.,
        w2_noise_dist: float = 0.,
        **kwargs):

    # stores
    w2_p1__q1_store = {-1: dict(w2_p1__q1_global_lipschitz=w2_p__q, w2_p1__q1_lagrangian_duality=w2_p__q)}
    w2_q__disc_q_store = dict()
    q_store = {-1: {'q1': q}}

    # initialize empirical distributions
    samples_store = {-1: {'p1_samples': sample_from_ambiguity_set(q, w2_p__q, num_samples),
                          'q1_samples': sample_from_ambiguity_set(q, 0., num_samples)}}

    # loop over time steps
    for k in range(num_time_steps):
        print(f'---- TIME STEP {k} ----')
        out = single_step(
            dynamics=dynamics,
            noise_dist=noise_dist,
            q=q_store[k-1]['q1'],
            p_samples=samples_store[k-1]['p1_samples'],
            w2_p__q_global_lipschitz=w2_p1__q1_store[k-1]['w2_p1__q1_global_lipschitz'],
            w2_p__q_lagrangian_duality=w2_p1__q1_store[k-1]['w2_p1__q1_lagrangian_duality'],
            w2_noise_dist=w2_noise_dist,
            num_samples=num_samples,
            **kwargs
        )
        w2_p1__q1_store[k] = {key: value for key, value in out.items() if 'w2_p1__q1' in key}
        w2_q__disc_q_store[k] = out['w2_q__disc_q']
        samples_store[k] = {key: value for key, value in out.items() if 'samples' in key}
        q_store[k] = dict(q1=out['q1'], q_compr=out['q_comp'])

        print(
            f"Bounds on W_2(p_{k+1}, q_{k+1}) via:\n"
            f"\t Global Lipschitz: {out['w2_p1__q1_global_lipschitz']:.4f}\n"
            f"\t Empirical: {out['w2_p1__q1_empirical']:.4f}\n"
            f"\t Lagrangian Duality: {out['w2_p1__q1_lagrangian_duality']:.4f}\n"
        )

    return w2_q__disc_q_store, w2_p1__q1_store, samples_store, q_store
