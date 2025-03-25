import torch
import discretize_distributions as ds
from typing import Union
from copy import copy
import GMMWas


def get_initial_dist(loc_initial_dist: torch.Tensor, variance_initial_dist: torch.Tensor):
    return construct_diag_gaussian_dist(loc_initial_dist, variance_initial_dist)


def get_noise_dist(loc_noise_dist: torch.Tensor, variance_noise_dist: torch.Tensor):
    return construct_diag_gaussian_dist(loc_noise_dist, variance_noise_dist)


def construct_diag_gaussian_dist(loc_dist: Union[list, torch.Tensor], variance_dist: Union[list, torch.Tensor]):
    loc_dist = torch.as_tensor(loc_dist)
    covariance_dist = torch.diag(torch.as_tensor(variance_dist))
    return ds.MultivariateNormal(loc=loc_dist, covariance_matrix=covariance_dist)


def cross_product(state_signature, noise_signature):

    n, m = state_signature.locs.size(0), noise_signature.locs.size(0)
    d, q = state_signature.locs.shape[-1], noise_signature.locs.shape[-1]

    cross_locs = torch.cat((
        state_signature.locs.unsqueeze(1).expand(-1, m, -1),
        noise_signature.locs.unsqueeze(0).expand(n, -1, -1)),
        dim=-1).view(-1, d + q)

    cross_probs = ( state_signature.probs.unsqueeze(1) * noise_signature.probs.unsqueeze(0) ).view(-1)

    return cross_probs, cross_locs

def sum_discrete_distributions(state_signature, noise_signature):

    d = state_signature.locs.size(-1)

    sum_locs = ( state_signature.locs.unsqueeze(1) + noise_signature.locs.unsqueeze(0) ).view(-1, d)
    sum_probs = ( state_signature.probs.unsqueeze(1) * noise_signature.probs.unsqueeze(0) ).view(-1)

    return sum_probs, sum_locs

@torch.no_grad()
def compress(
        q: Union[ds.MultivariateNormal, ds.MixtureMultivariateNormal, ds.CategoricalFloat],
        num_locs_after_compr: int
):
    if isinstance(q, ds.MultivariateNormal) or num_locs_after_compr >= q.num_components:
        w2_compr = 0.
    else:
        if isinstance(q, ds.MixtureMultivariateNormal):
            if (q.component_distribution.covariance_matrix == q.component_distribution.covariance_matrix[0]).all():
                # We restrict the compressed distribution to a mixture with each component the covariance_matrix
                # of all components of the q
                q_core_org = ds.CategoricalFloat(probs=q.mixture_distribution.probs, locs=q.component_distribution.mean)
                q_core = ds.compress_categorical_floats(q_core_org, n_max=num_locs_after_compr)
                w2_compr = GMMWas.w2(q_core, q_core_org)
                q = ds.MixtureMultivariateNormal(
                    mixture_distribution=torch.distributions.Categorical(probs=q_core.probs),
                    component_distribution=ds.MultivariateNormal(
                        loc=q_core.locs, covariance_matrix=q.component_distribution.covariance_matrix[0]))
            else:
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
    return q, w2_compr