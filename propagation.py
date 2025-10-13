import torch
from typing import Union
import discretize_distributions as ds
from dynamics import Dynamics, AdditiveGaussianDynamics
from utils_distributions import cross_product, sum_discrete_distributions


def propagate_additive_gaussian_noise(
        dynamics: AdditiveGaussianDynamics,
        noise_dist: ds.MultivariateNormal,
        sign_state_dist: ds.CategoricalFloat
):
    assert isinstance(dynamics, AdditiveGaussianDynamics) and isinstance(noise_dist, ds.MultivariateNormal), \
        ValueError('Only supports additive Gaussian noise')

    return ds.MixtureMultivariateNormal(
        mixture_distribution=torch.distributions.Categorical(
            probs=sign_state_dist.probs),
        component_distribution=ds.MultivariateNormal(
            loc=dynamics.state_dynamics(sign_state_dist.locs) + noise_dist.loc,
            covariance_matrix=noise_dist.covariance_matrix
        ))


def propagate_additive_discrete_noise(
        dynamics: AdditiveGaussianDynamics, # \todo works for general AdditiveDynamics
        sign_noise_dist: ds.CategoricalFloat,
        sign_state_dist: ds.CategoricalFloat
):
    assert isinstance(dynamics, AdditiveGaussianDynamics), ValueError('Only supports additive noise')
    propagated_states = ds.CategoricalFloat(
        probs=sign_state_dist.probs, locs=dynamics.state_dynamics(sign_state_dist.locs))
    sum_probs, sum_locs = sum_discrete_distributions(propagated_states, sign_noise_dist)
    q1 = ds.CategoricalFloat(probs=sum_probs, locs=sum_locs)

    return q1


def propagate_general_discrete_noise(
        dynamics: Dynamics,
        sign_noise_dist: ds.CategoricalFloat,
        sign_state_dist: ds.CategoricalFloat
):
    cross_probs, cross_locs = cross_product(sign_state_dist, sign_noise_dist)
    sign_cross = ds.CategoricalFloat(probs=cross_probs, locs=cross_locs)
    q1 = ds.CategoricalFloat(probs=cross_probs, locs=dynamics(cross_locs))

    return sign_cross, q1
