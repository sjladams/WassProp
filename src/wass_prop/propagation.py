import torch
from dataclasses import dataclass, field
from typing import Union, List, Optional, Tuple, Dict

import discretize_distributions.distributions as dd_dists

from . import wasserstein
from .dynamics import StochasticDynamics, AdditiveNoiseDynamics
from .utils_distributions import AmbiguityBall, discretize, cross_product, sum_discrete_distributions

TensorLike = Union[float, torch.Tensor]


def single_step(
    dynamics: StochasticDynamics,
    q: AmbiguityBall,
    noise: AmbiguityBall,
    num_locs: int,
    use_lagrangian_duality: bool = True,
    use_additive_noise: bool = True,
) -> AmbiguityBall:
    if isinstance(dynamics, AdditiveNoiseDynamics) and use_additive_noise:
        return _single_step_additive_noise(dynamics, q, noise, num_locs, use_lagrangian_duality)
    else:
        return _single_step_general_noise(dynamics, q, noise, num_locs, use_lagrangian_duality)

def _single_step_general_noise(
    dynamics: StochasticDynamics,
    q: AmbiguityBall,
    noise: AmbiguityBall,
    num_locs: int,
    use_lagrangian_duality: bool = True,
) -> AmbiguityBall:
    disc_q, w2_q__disc_q = discretize(q, num_locs)
    disc_noise_dist, w2_noise_dist__disc_noise_dist = discretize(noise, num_locs)

    if isinstance(dynamics, AdditiveNoiseDynamics):
        q1 = propagate_additive_noise(dynamics, disc_noise_dist, disc_q)
    else:
        q1 = propagate_general_discrete_noise(dynamics, disc_noise_dist, disc_q)

    if isinstance(dynamics, AdditiveNoiseDynamics):
        if use_lagrangian_duality:
            w2_p__q = q.w2

            if isinstance(q.center, dd_dists.MultivariateNormal) and w2_p__q == 0.:
                w2_p1__q1 = wasserstein.compute_w2_f_q__f_disc_q_lagrangian_duality(
                    q=q.center,
                    disc_q=disc_q,
                    f=dynamics.state_dynamics,
                )
            else:
                w2_p1__q1 = wasserstein.compute_w2_f_p__f_disc_q_lagrangian_duality(
                    disc_q=disc_q,
                    f=dynamics.state_dynamics,
                    w2_p__disc_q=w2_p__q + w2_q__disc_q,
                )

            w2_p__q += w2_noise_dist__disc_noise_dist + noise.w2
        else:
            w2_p1__q1 = dynamics.global_lipschitz * (w2_q__disc_q + q.w2) + w2_noise_dist__disc_noise_dist + noise.w2

    else:
        if use_lagrangian_duality:
            w2_p__q = q.w2 + noise.w2
            w2_q__disc_q = w2_q__disc_q + w2_noise_dist__disc_noise_dist

            if isinstance(q.center, dd_dists.MultivariateNormal) and w2_p__q == 0.:
                w2_p1__q1 = wasserstein.compute_w2_f_q__f_disc_q_lagrangian_duality(
                    q=q.center,
                    disc_q=disc_q,
                    f=dynamics,
                )
            else:
                w2_p1__q1 = wasserstein.compute_w2_f_p__f_disc_q_lagrangian_duality(
                    disc_q=disc_q,
                    f=dynamics,
                    w2_p__disc_q=w2_p__q + w2_q__disc_q
                )
        else:
            w2_p1__q1 = dynamics.global_lipschitz * (w2_q__disc_q + q.w2 + w2_noise_dist__disc_noise_dist + noise.w2)

    return AmbiguityBall(center=q1, radius=w2_p1__q1)

def _single_step_additive_noise(
    dynamics: AdditiveNoiseDynamics,
    q: AmbiguityBall,
    noise: AmbiguityBall,
    num_locs: int,
    use_lagrangian_duality: bool = True,
) -> AmbiguityBall:
    disc_q, w2_q__disc_q = discretize(q, num_locs)

    q1 = propagate_additive_noise(dynamics, noise.center, disc_q)

    if use_lagrangian_duality:
        w2_p__q = q.w2

        if isinstance(q.center, dd_dists.MultivariateNormal) and w2_p__q == 0.:
            w2_p1__q1 = wasserstein.compute_w2_f_q__f_disc_q_lagrangian_duality(
                q=q.center,
                disc_q=disc_q,
                f=dynamics.state_dynamics,
            )
        else:
            w2_p1__q1 = wasserstein.compute_w2_f_p__f_disc_q_lagrangian_duality(
                disc_q=disc_q,
                f=dynamics.state_dynamics,
                w2_p__disc_q=w2_p__q + w2_q__disc_q,
            )
    else:
        w2_p1__q1 = dynamics.global_lipschitz * (w2_q__disc_q + q.w2)

    w2_p1__q1 += noise.w2

    return AmbiguityBall(center=q1, radius=w2_p1__q1)

@dataclass
class Path:
    steps: Dict[int, AmbiguityBall] = field(default_factory=dict)

    def append(self, k: int, rec: AmbiguityBall) -> None:
        self.steps[k] = rec

    def at(self, k: int) -> AmbiguityBall:
        return self.steps[k]

    @property
    def ordered_indices(self) -> List[int]:
        return sorted(self.steps.keys())

def multi_step(
    dynamics: StochasticDynamics,
    q: AmbiguityBall,
    noise: AmbiguityBall,
    num_time_steps: int,
    num_locs: int,
    use_lagrangian_duality: bool = True,
    use_additive_noise: bool = True,
    print_progress: bool = True,
) -> Path:

    path = Path()
    path.append(-1, q)

    for k in range(num_time_steps):
        path.append(k, single_step(
            dynamics=dynamics,
            q=path.at(k-1),
            noise=noise,
            num_locs=num_locs,
            use_lagrangian_duality=use_lagrangian_duality, 
            use_additive_noise=use_additive_noise,
        ))
        if print_progress:
            print(f"W_2(p_{k+1}, q_{k+1}) <= {path.at(k).w2:.4f}:\n")

    return path

def single_step_empirical(
    dynamics: StochasticDynamics,
    p_emp: torch.Tensor,
    noise: AmbiguityBall,
    num_samples: int,
) -> torch.Tensor:
    noise_emp = noise.sample(num_samples)
    p1_emp = dynamics(torch.cat((p_emp, noise_emp), dim=-1))
    return p1_emp

@dataclass
class SampledPath:
    steps: Dict[int, torch.Tensor] = field(default_factory=dict)

    def append(self, k: int, rec: torch.Tensor) -> None:
        self.steps[k] = rec

    def at(self, k: int) -> torch.Tensor:
        return self.steps[k]

    @property
    def ordered_indices(self) -> List[int]:
        return sorted(self.steps.keys())

    def detach(self) -> "SampledPath":
        new = SampledPath()
        for k, v in self.steps.items():
            new.steps[k] = v.detach().clone() if isinstance(v, torch.Tensor) else v
        return new

def multi_step_empirical(
    dynamics: StochasticDynamics,
    p_emp: torch.Tensor,
    noise: AmbiguityBall,
    num_time_steps: int,
) -> SampledPath:

    path = SampledPath()
    path.append(-1, p_emp)

    for k in range(num_time_steps):
        path.append(k, single_step_empirical(
            dynamics=dynamics,
            p_emp=path.at(k-1),
            noise=noise,
            num_samples=p_emp.size(0)
        ))

    return path


def propagate_additive_noise(
        dynamics: AdditiveNoiseDynamics,
        noise_dist: Union[dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal, dd_dists.CategoricalFloat],
        disc_state_dist: dd_dists.CategoricalFloat
):
    if not isinstance(dynamics, AdditiveNoiseDynamics):
        raise ValueError('Only supports additive noise')

    if isinstance(noise_dist, dd_dists.MultivariateNormal):
        return dd_dists.MixtureMultivariateNormal(
            mixture_distribution=torch.distributions.Categorical(
                probs=disc_state_dist.probs),
            component_distribution=dd_dists.MultivariateNormal(
                loc=dynamics.state_dynamics(disc_state_dist.locs) + noise_dist.loc,
                covariance_matrix=noise_dist.covariance_matrix
            ))
    elif isinstance(noise_dist, dd_dists.MixtureMultivariateNormal):
        probs, locs, covs = list(), list(), list()
        for i in range(noise_dist.num_components):
            probs.append(disc_state_dist.probs * noise_dist.mixture_distribution.probs[i])
            locs.append(dynamics.state_dynamics(disc_state_dist.locs) + noise_dist.component_distribution.loc[i])
            covs.append(noise_dist.component_distribution.covariance_matrix[i].expand(disc_state_dist.num_components, -1, -1))

        return dd_dists.MixtureMultivariateNormal(
            mixture_distribution=torch.distributions.Categorical(probs=torch.cat(probs)),
            component_distribution=dd_dists.MultivariateNormal(
                loc=torch.cat(locs),
                covariance_matrix=torch.cat(covs)
            ))
    elif isinstance(noise_dist, dd_dists.CategoricalFloat):
        propagated_states = dd_dists.CategoricalFloat(
            probs=disc_state_dist.probs, 
            locs=dynamics.state_dynamics(disc_state_dist.locs)
        )
        sum_probs, sum_locs = sum_discrete_distributions(propagated_states, noise_dist)
        return dd_dists.CategoricalFloat(probs=sum_probs, locs=sum_locs)
    else:
        raise NotImplementedError(f"Noise of type {type(noise_dist)} not supported")


def propagate_general_discrete_noise(
        dynamics: StochasticDynamics,
        sign_noise_dist: dd_dists.CategoricalFloat,
        disc_state_dist: dd_dists.CategoricalFloat
):
    cross_probs, cross_locs = cross_product(disc_state_dist, sign_noise_dist)
    return dd_dists.CategoricalFloat(probs=cross_probs, locs=dynamics(cross_locs))
