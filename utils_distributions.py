import torch
import discretize_distributions as ds
from typing import Union, List, Optional
import wasserstein
from regions import HyperRectangularVoronoiPartition

def get_initial_dist(loc_initial_dist, variance_initial_dist, **kwargs):
    return construct_diag_gaussian_dist(loc_initial_dist, variance_initial_dist)


def get_noise_dist(loc_noise_dist, variance_noise_dist, **kwargs):
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

def quantize(q, num_locs):

    if isinstance(q, ds.MultivariateNormal):
        sign_q = ds.discretization_generator(dist=q, num_locs=num_locs)
        w2_quantization = sign_q.w2

    elif isinstance(q, ds.DiscretizedMultivariateNormal) or isinstance(q, ds.CategoricalFloat):
        sign_q = q
        w2_quantization = 0.0 # TODO: Assuming same support, this needs to be extended

    else:
        raise ValueError('Optimal quantization not implemented for q class.')

    return sign_q, w2_quantization