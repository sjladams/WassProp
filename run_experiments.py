import torch
import matplotlib.pyplot as plt

from setup_experiments import multi_step_uq
from dynamics import GaussianDynamics1d, ChaoticDynamics, LinearDynamics

if __name__ == '__main__':
    torch.manual_seed(0) # for reproducibility

    # Parameter Settings # \todo import as Namespace

    # dynamics = GaussianDynamics1d
    # num_dims = 1
    # params_dynamics = {'loc': torch.zeros(num_dims), 'scale': torch.ones(num_dims)}

    dynamics = ChaoticDynamics
    num_dims = 1
    params_dynamics = {'r': 4}

    # dynamics = LinearDynamics
    # num_dims = 1
    # params_dynamics = {'mat': torch.tensor([[0.5]])}

    # dynamics = LinearDynamics
    # num_dims = 2
    # params_dynamics = {'mat': torch.diag(torch.Tensor([0.1, 0.9]))}

    params_noise_dist = {'loc': torch.zeros(num_dims), 'covariance_matrix': torch.diag(torch.ones(num_dims) * 0.3 ** 2)}
    # params_noise_dist = {'loc': torch.zeros(num_dims), 'covariance_matrix': torch.diag(torch.Tensor([0.01, 0.01]))}
    params_initial_dist = {'loc': torch.zeros(num_dims), 'covariance_matrix': torch.eye(num_dims)}
    # params_initial_dist = {'loc': torch.zeros(num_dims), 'covariance_matrix': torch.diag(torch.Tensor([0.01, 0.01]))}

    lipschitz = dynamics.global_lipschitz
    params_simulation = {'num_samples': 1000, 'K': 5, 'plot': False}
    params_signature = {'nr_signature_points': 12}


    # Run the experiment
    w2_bounds, tag = multi_step_uq(dynamics, params_dynamics, params_noise_dist, params_initial_dist, params_signature,
                                   params_simulation)

    # Plot the results
    fig_w2_bounds = plt.figure()
    plt.plot(range(params_simulation['K']+1), w2_bounds['emp'], label='Empirical')
    plt.plot(range(params_simulation['K']+1), w2_bounds['gl'], label='Global Lipschitz')
    plt.plot(range(params_simulation['K']+1), w2_bounds['type1'],
             label=r'Own (Budget Term 2 = $W_2(\Delta p,\Delta q)$)')
    plt.plot(range(params_simulation['K'] + 1), w2_bounds['type2'],
             label=r'Own (Budget Term 2 = $W_2(p,\Delta q)$)')
    plt.legend()
    plt.title(tag)
    plt.xticks(range(params_simulation['K'] + 1))
    plt.xlabel('k')
    plt.ylabel(r'$W_2(p_k, q_k)$')

    if dynamics.__name__ == 'ChaoticDynamics':
        plt.yscale('log')
        plt.xlim(1, params_simulation['K'])
    else:
        plt.xlim(0, params_simulation['K'])

    plt.show()

    if dynamics.__name__ in ['ChaoticDynamics', 'LinearDynamics']:
        fig_our_w2_bounds = plt.figure()
        plt.plot(range(params_simulation['K'] + 1), w2_bounds['type1'],
                 label=r'Own (Budget Term 2 = $W_2(\Delta p,\Delta q)$)')
        plt.plot(range(params_simulation['K'] + 1), w2_bounds['type2'],
                 label=r'Own (Budget Term 2 = $W_2(p,\Delta q)$)')
        plt.legend()
        plt.title(tag)
        plt.xticks(range(params_simulation['K'] + 1))
        plt.xlabel('k')
        plt.ylabel(r'$W_2(p_k, q_k)$')

        plt.xlim(0, params_simulation['K'])

        plt.show()
