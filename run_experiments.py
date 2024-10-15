import torch

from setup_experiments import multi_step_uq
from dynamics import GaussianDynamics1d, ChaoticDynamics

if __name__ == '__main__':
    torch.manual_seed(0) # for reproducibility

    # Parameter Settings # \todo import as Namespace
    num_dims = 1

    # dynamics = GaussianDynamics1d
    # params_dynamics = {'loc': torch.zeros(num_dims), 'scale': torch.ones(num_dims)}

    dynamics = ChaoticDynamics
    params_dynamics = {'r': 4}

    params_noise_dist = {'loc': torch.zeros(num_dims), 'covariance_matrix': torch.diag(torch.ones(num_dims) * 0.3 ** 2)}
    params_initial_dist = {'loc': torch.zeros(num_dims), 'covariance_matrix': torch.eye(num_dims)}
    params_simulation = {'num_samples': 1000, 'K': 3}
    params_signature = {'nr_signature_points': 10}

    multi_step_uq(dynamics, params_dynamics, params_noise_dist, params_initial_dist, params_signature, params_simulation)



