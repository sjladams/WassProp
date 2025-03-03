import torch
from typing import Union
import discretize_distributions as ds
from dynamics import Dynamics, AdditiveGaussianDynamics
from utils_distributions import cross_product, sum_discrete_distributions

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
        propagate_via_gmm: bool
):
    if propagate_via_gmm:
        sign_q, q1 = _propagate_via_gmms(dynamics, noise_dist, sign_state_dist)
    else:
        if isinstance(dynamics, AdditiveGaussianDynamics):
            sign_q, q1 = _propagate_state_noise_discrete_additive(dynamics, noise_dist, sign_state_dist)

        else:
            sign_q, q1 = _propagate_state_noise_discrete_general(dynamics, noise_dist, sign_state_dist)

    return sign_q, q1