import torch
from typing import Union, Tuple
from copy import copy
import ot

import discretize_distributions.distributions as dd_dists
import discretize_distributions as dd

Dist = Union[dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal, dd_dists.CategoricalFloat]

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

DiscreteDist = Union[torch.Tensor, dd_dists.CategoricalFloat]


def w2_discrete(p: DiscreteDist, q: DiscreteDist) -> torch.Tensor:
    """2-Wasserstein distance between two discrete distributions, each given either as raw
    samples (uniform weights assumed) or as a CategoricalFloat (locs + probs)."""
    p_locs, p_probs = (p.locs, p.probs) if isinstance(p, dd_dists.CategoricalFloat) else (p, None)
    q_locs, q_probs = (q.locs, q.probs) if isinstance(q, dd_dists.CategoricalFloat) else (q, None)
    return ot.solve_sample(X_a=p_locs, X_b=q_locs, a=p_probs, b=q_probs, metric="sqeuclidean").value.sqrt()


@torch.no_grad()
def compress_categorical_float(
        q: dd_dists.CategoricalFloat,
        size_after_compr: int
) -> Tuple[dd_dists.CategoricalFloat, Union[float, torch.Tensor]]:
    if size_after_compr >= q.num_components:
        w2_compr = 0.
    else:
        q_pre = copy(q)
        q = dd_dists.compress_categorical_floats(q_pre, n_max=size_after_compr)
        w2_compr = w2_discrete(q, q_pre)
    return q, w2_compr

class AmbiguityBall:
    def __init__(
            self, 
            center: Dist, 
            radius: Union[float, torch.Tensor]
        ):
        self.center = center
        self.radius = radius

    @property
    def w2(self):
        return self.radius

    def sample(self, num_samples: int):
        if self.radius == 0. or (isinstance(self.radius, torch.Tensor) and self.radius.isnan().any()): # TODO make this explicit
            return self.center.sample(torch.Size((num_samples,)))
        else:  # TODO fix issue num_samples != sqrt(num_samples) ** 2
            assert isinstance(self.center, (dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal,
                                             dd_dists.CategoricalFloat)), (
                ValueError('Only implemented for (mixtures of) MultivariateNormal and CategoricalFloat distributions'))

            # sample sqrt(num_samples) vectors from standard normal distribution
            vec = torch.randn(int(num_samples**0.5), self.center.mean.shape[-1])

            # scale vectors to have length w2
            vec = (vec / vec.norm(dim=1, keepdim=True)) * self.radius

            # sample radii
            r = torch.rand(vec.shape[0]).pow(1 / self.center.mean.shape[-1]).unsqueeze(1)

            # scale vectors by radii
            vec = r * vec

            # create sqrt(num_samples) distributions of type center with the means perturbed by the scaled vectors
            if isinstance(self.center, dd_dists.MultivariateNormal):
                perturbed_center = dd_dists.MultivariateNormal(
                    loc=self.center.mean.unsqueeze(-2) + vec,
                    covariance_matrix=self.center.covariance_matrix
                )
            elif isinstance(self.center, dd_dists.MixtureMultivariateNormal):
                weighted_vec = vec.unsqueeze(-2).expand(-1, self.center.num_components, -1) * self.center.mixture_distribution.probs.unsqueeze(0).unsqueeze(-1)
                perturbed_center = dd_dists.MixtureMultivariateNormal(
                    mixture_distribution=torch.distributions.Categorical(
                        probs=self.center.mixture_distribution.probs.unsqueeze(0).expand(vec.shape[0], -1)),
                    component_distribution=dd_dists.MultivariateNormal(
                        loc=self.center.component_distribution.mean.unsqueeze(-3) + weighted_vec,
                        covariance_matrix=self.center.component_distribution.covariance_matrix))
            elif isinstance(self.center, dd_dists.CategoricalFloat):
                weighted_vec = vec.unsqueeze(-2).expand(-1, self.center.num_components, -1) * self.center.probs.unsqueeze(0).unsqueeze(-1)
                perturbed_center = dd_dists.CategoricalFloat(
                    locs=self.center.locs.unsqueeze(-3) + weighted_vec,
                    probs=self.center.probs.unsqueeze(0).expand(vec.shape[0], -1)
                )
            else:
                raise NotImplementedError

            # take sqrt(num_samples) samples from perturbed distributions
            samples = perturbed_center.sample(torch.Size((int(num_samples**0.5),)))
            return samples.flatten(start_dim=-3, end_dim=-2)

def discretize(q: Dist, num_locs: int, configuration: str = 'grid') -> Tuple[dd_dists.CategoricalFloat, Union[torch.Tensor, float]]:
    if isinstance(q, dd_dists.CategoricalFloat):
        return compress_categorical_float(q, size_after_compr=num_locs)
    else:
        if configuration == 'cross':
            per_mode = False
        else:
            per_mode = True

        scheme_q = dd.generate_scheme(dist=q, scheme_size=num_locs, configuration=configuration, per_mode=per_mode)
        return dd.discretize(q, scheme_q)
