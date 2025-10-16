from typing import Union, List
import torch
import discretize_distributions.distributions as dd_dists


def get_initial_dist(
    loc: Union[torch.Tensor, List[float]], 
    variance: Union[torch.Tensor, List[float]]
):
    return construct_diag_gaussian_dist(loc, variance)


def get_noise_dist(
    loc: Union[torch.Tensor, List[float]], 
    variance: Union[torch.Tensor, List[float]]
):
    if all(isinstance(i, list) for i in loc) and all(isinstance(i, list) for i in variance):
        return dd_dists.MixtureMultivariateNormal(
            mixture_distribution=torch.distributions.Categorical(probs=torch.ones(len(loc))),
            component_distribution=dd_dists.MultivariateNormal(
                loc=torch.as_tensor(loc),
                covariance_matrix=torch.diag_embed(torch.as_tensor(variance))))
    else:
        return construct_diag_gaussian_dist(loc, variance)


def construct_diag_gaussian_dist(
    loc: Union[list, torch.Tensor], 
    variance: Union[list, torch.Tensor]
):
    return dd_dists.MultivariateNormal(
        loc=torch.as_tensor(loc), 
        covariance_matrix=torch.diag(torch.as_tensor(variance))
    )


def rot_mat(theta, rho, delta):
    theta = torch.as_tensor(theta)
    rho = torch.as_tensor(rho)
    delta = torch.as_tensor(delta)
    return rho * torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]) + delta
