import torch
from dataclasses import dataclass, field
from typing import Union, List, Optional, Tuple, Dict

import discretize_distributions.distributions as dd_dists

from . import wasserstein
from .dynamics import StochasticDynamics, AdditiveNoiseDynamics
from .utils_distributions import AmbiguityBall, discretize, cross_product, sum_discrete_distributions

TensorLike = Union[float, torch.Tensor]


@dataclass
class SingleStepConfig:
    q: AmbiguityBall
    noise: AmbiguityBall
    num_locs: int
    use_lagrangian_duality: bool = True

def single_step(
    dynamics: StochasticDynamics,
    q: AmbiguityBall,
    noise: AmbiguityBall,
    num_locs: int,
    use_lagrangian_duality: bool = True,
) -> AmbiguityBall:
    cfg = SingleStepConfig(
        q=q,
        noise=noise,
        num_locs=num_locs,
        use_lagrangian_duality=use_lagrangian_duality
    )
    if isinstance(dynamics, AdditiveNoiseDynamics):
        return _single_step_additive_noise(dynamics, cfg)
    else:
        return _single_step_general_noise(dynamics, cfg)

def _single_step_general_noise(
    dynamics: StochasticDynamics,
    cfg: SingleStepConfig
) -> AmbiguityBall:
    disc_q, w2_q__disc_q = discretize(cfg.q, cfg.num_locs)
    disc_noise_dist, w2_noise_dist__disc_noise_dist = discretize(cfg.noise, cfg.num_locs)

    q1 = propagate_general_discrete_noise(dynamics, disc_noise_dist, disc_q)

    if cfg.use_lagrangian_duality:
        w2_p1__q1 = wasserstein.compute_w2_f_p__f_disc_q_lagrangian_duality(
            disc_q=disc_q,
            f=dynamics, 
            w2_p__disc_q=w2_q__disc_q + w2_noise_dist__disc_noise_dist + cfg.q.w2 + cfg.noise.w2
        )
    else:
        w2_p1__q1 = dynamics.global_lipschitz * (w2_q__disc_q + cfg.q.w2 + w2_noise_dist__disc_noise_dist + cfg.noise.w2)

    return AmbiguityBall(center=q1, radius=w2_p1__q1)

def _single_step_additive_noise(
    dynamics: AdditiveNoiseDynamics,
    cfg: SingleStepConfig
) -> AmbiguityBall:
    disc_q, w2_q__disc_q = discretize(cfg.q, cfg.num_locs)

    q1 = propagate_additive_gaussian_noise(dynamics, cfg.noise.center, disc_q)

    if cfg.use_lagrangian_duality:
        w2_p1__q1 = wasserstein.compute_w2_f_p__f_disc_q_lagrangian_duality(
            disc_q=disc_q, 
            f=dynamics.state_dynamics, 
            w2_p__disc_q=w2_q__disc_q + cfg.q.w2
        )
    else:
        w2_p1__q1 = dynamics.global_lipschitz * (w2_q__disc_q + cfg.q.w2)

    w2_p1__q1 += cfg.noise.w2

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
) -> Path:

    path = Path()
    path.append(-1, q)

    for k in range(num_time_steps):
        print(f'---- TIME STEP {k} ----')
        path.append(k, single_step(
            dynamics=dynamics,
            q=path.at(k-1),
            noise=noise,
            num_locs=num_locs,
            use_lagrangian_duality=use_lagrangian_duality
        ))

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

    def ordered_indices(self) -> List[int]:
        return sorted(self.steps.keys())

def multi_step_empirical(
    dynamics: StochasticDynamics,
    p_emp: torch.Tensor,
    noise: AmbiguityBall,
    num_time_steps: int,
) -> SampledPath:

    path = SampledPath()
    path.append(-1, p_emp)

    for k in range(num_time_steps):
        print(f'---- TIME STEP {k} ----')
        path.append(k, single_step_empirical(
            dynamics=dynamics,
            p_emp=path.at(k-1),
            noise=noise,
            num_samples=p_emp.size(0)
        ))

    return path


def propagate_additive_gaussian_noise(
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
