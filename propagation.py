import torch
from typing import Union
import discretize_distributions.distributions as dd_dists
from dynamics import Dynamics, AdditiveGaussianDynamics
from utils_distributions import cross_product, sum_discrete_distributions


def propagate_additive_gaussian_noise(
        dynamics: AdditiveGaussianDynamics,
        noise_dist: dd_dists.MultivariateNormal,
        sign_state_dist: dd_dists.CategoricalFloat
):
    assert (isinstance(dynamics, AdditiveGaussianDynamics) and
            isinstance(noise_dist, (dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal))), (
        ValueError('Only supports additive (mixtures of) Gaussian noise'))

    if isinstance(noise_dist, dd_dists.MultivariateNormal):
        return dd_dists.MixtureMultivariateNormal(
            mixture_distribution=torch.distributions.Categorical(
                probs=sign_state_dist.probs),
            component_distribution=dd_dists.MultivariateNormal(
                loc=dynamics.state_dynamics(sign_state_dist.locs) + noise_dist.loc,
                covariance_matrix=noise_dist.covariance_matrix
            ))
    elif isinstance(noise_dist, dd_dists.MixtureMultivariateNormal):
        probs, locs, covs = list(), list(), list()
        for i in range(noise_dist.num_components):
            probs.append(sign_state_dist.probs * noise_dist.mixture_distribution.probs[i])
            locs.append(dynamics.state_dynamics(sign_state_dist.locs) + noise_dist.component_distribution.loc[i])
            covs.append(noise_dist.component_distribution.covariance_matrix[i].expand(sign_state_dist.num_components, -1, -1))

        return dd_dists.MixtureMultivariateNormal(
            mixture_distribution=torch.distributions.Categorical(probs=torch.cat(probs)),
            component_distribution=dd_dists.MultivariateNormal(
                loc=torch.cat(locs),
                covariance_matrix=torch.cat(covs)
            ))


def propagate_additive_discrete_noise(
        dynamics: AdditiveGaussianDynamics, # \todo works for general AdditiveDynamics
        sign_noise_dist: dd_dists.CategoricalFloat,
        sign_state_dist: dd_dists.CategoricalFloat
):
    assert isinstance(dynamics, AdditiveGaussianDynamics), ValueError('Only supports additive noise')
    propagated_states = dd_dists.CategoricalFloat(
        probs=sign_state_dist.probs, locs=dynamics.state_dynamics(sign_state_dist.locs))
    sum_probs, sum_locs = sum_discrete_distributions(propagated_states, sign_noise_dist)
    q1 = dd_dists.CategoricalFloat(probs=sum_probs, locs=sum_locs)

    return q1


def propagate_general_discrete_noise(
        dynamics: Dynamics,
        sign_noise_dist: dd_dists.CategoricalFloat,
        sign_state_dist: dd_dists.CategoricalFloat
):
    cross_probs, cross_locs = cross_product(sign_state_dist, sign_noise_dist)
    sign_cross = dd_dists.CategoricalFloat(probs=cross_probs, locs=cross_locs)
    q1 = dd_dists.CategoricalFloat(probs=cross_probs, locs=dynamics(cross_locs))

    return q1
