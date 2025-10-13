import torch
import discretize_distributions.distributions as dd_dists
from typing import Union
from copy import copy
import ot


def get_initial_dist(loc_initial_dist: torch.Tensor, variance_initial_dist: torch.Tensor):
    return construct_diag_gaussian_dist(loc_initial_dist, variance_initial_dist)


def get_noise_dist(loc_noise_dist: torch.Tensor, variance_noise_dist: torch.Tensor):
    if all(isinstance(i, list) for i in loc_noise_dist) and all(isinstance(i, list) for i in variance_noise_dist):
        return dd_dists.MixtureMultivariateNormal(
            mixture_distribution=torch.distributions.Categorical(probs=torch.ones(len(loc_noise_dist))),
            component_distribution=dd_dists.MultivariateNormal(
                loc=torch.as_tensor(loc_noise_dist),
                covariance_matrix=torch.diag_embed(torch.as_tensor(variance_noise_dist))))
    else:
        return construct_diag_gaussian_dist(loc_noise_dist, variance_noise_dist)


def construct_diag_gaussian_dist(loc_dist: Union[list, torch.Tensor], variance_dist: Union[list, torch.Tensor]):
    loc_dist = torch.as_tensor(loc_dist)
    covariance_dist = torch.diag(torch.as_tensor(variance_dist))
    return dd_dists.MultivariateNormal(loc=loc_dist, covariance_matrix=covariance_dist)


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

    sum_locs = ( state_signature.locs.unsqueeze(1) + noise_signature.locs.unsqueeze(0) ).reshape(-1, d)
    sum_probs = ( state_signature.probs.unsqueeze(1) * noise_signature.probs.unsqueeze(0) ).view(-1)

    return sum_probs, sum_locs

@torch.no_grad()
def compress_compress_categorical_floats(
        q: dd_dists.CategoricalFloat,
        size_after_compr: int
):
    if size_after_compr >= q.num_components:
        w2_compr = 0.
    else:
        q_pre = copy(q)
        q = dd_dists.compress_categorical_floats(q_pre, n_max=size_after_compr)
        w2_compr = ot.solve_sample(X_a=q.locs, a=q.probs, X_b=q_pre.locs, b=q_pre.probs).value.sqrt()
    return q, w2_compr


def sample_from_ambiguity_set(center: Union[dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal], w2: float, num_samples: int):
    if w2 == 0.:
        return center.sample(torch.Size((num_samples,)))
    else:
        assert isinstance(center, (dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal)), (
            ValueError('Only implemented for (mixtures of) MultivariateNormal distributions'))

        # sample sqrt(num_samples) vectors from standard normal distribution
        vec = torch.randn(int(num_samples**0.5), center.mean.shape[-1])

        # scale vectors to have length w2
        vec = (vec / vec.norm(dim=1, keepdim=True)) * w2

        # sample radii
        r = torch.rand(vec.shape[0]).pow(1 / center.mean.shape[-1]).unsqueeze(1)

        # scale vectors by radii
        vec = r * vec

        # create sqrt(num_samples) distributions of type center with the means perturbed by the scaled vectors
        if isinstance(center, dd_dists.MultivariateNormal):
            perturbed_center = dd_dists.MultivariateNormal(
                loc=center.mean.unsqueeze(-2) + vec,
                covariance_matrix=center.covariance_matrix
            )
        elif isinstance(center, dd_dists.MixtureMultivariateNormal):
            weighted_vec = vec.unsqueeze(-2).expand(-1, center.num_components, -1) * center.mixture_distribution.probs.unsqueeze(0).unsqueeze(-1)
            perturbed_center = dd_dists.MixtureMultivariateNormal(
                mixture_distribution=torch.distributions.Categorical(
                    probs=center.mixture_distribution.probs.unsqueeze(0).expand(vec.shape[0], -1)),
                component_distribution=dd_dists.MultivariateNormal(
                    loc=center.component_distribution.mean.unsqueeze(-3) + weighted_vec,
                    covariance_matrix=center.component_distribution.covariance_matrix))
        else:
            raise NotImplementedError # \todo generalize to CategoricalFloat

        # take sqrt(num_samples) samples from perturbed distributions
        samples = perturbed_center.sample(torch.Size((int(num_samples**0.5),)))
        return samples.flatten(start_dim=-3, end_dim=-2)


