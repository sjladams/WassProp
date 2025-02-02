import torch
import discretize_distributions as ds

import wasserstein
from regions import HyperRectangularVoronoiPartition


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

def quantize(q, num_locs, locs = None):

    if locs is None:
        if isinstance(q, ds.MultivariateNormal):
            sign_q = ds.discretization_generator(dist=q, num_locs=num_locs)
            w2_quantization = sign_q.w2

        elif isinstance(q, ds.DiscretizedMultivariateNormal) or isinstance(q, ds.CategoricalFloat):
            sign_q = q
            w2_quantization = 0.0

        else:
            raise ValueError('Optimal quantization not implemented for q class.')

    else: # Quantize given locs using Voronoi partitions
        if isinstance(q, ds.MultivariateNormal):
            voronoi_partition = HyperRectangularVoronoiPartition(locs)
            compute_sq_norm_2nd_moment = wasserstein.compute_sq_norm_2nd_moment(q, voronoi_partition, locs)

            raise ValueError('NEED TO IMPLEMENT P(R_i) FOR GAUSSIAN.')

            #return torch.einsum('...i,...i->...', compute_sq_norm_2nd_moment, probs)

        else:
            raise ValueError('Quantization given locs not implemented for q class.')

    return sign_q, w2_quantization