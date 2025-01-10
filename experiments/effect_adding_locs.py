import torch

from experiments import multi_step, get_noise_dist, get_initial_dist, single_step_w2_options
from dynamics import get_dynamics
import plot

from utils import load_params, parse_arguments

def effect_adding_locs(dynamics_type, num_dims, dyn_setting, num_locs, w_p__q):
    args = parse_arguments(
        dynamics_type=dynamics_type,
        num_dims=num_dims,
        dynamics_setting=dyn_setting,
        num_locs=num_locs,
        num_locs_after_compr=1,
        num_samples=1000,
        lr=0.01,
        num_iterations=1000,
        plot=False,
        optimize_locs=False
    )

    run_single_step = True

    params = load_params(args)

    dynamics = get_dynamics(**params)
    initial_dist = get_initial_dist(**params)
    noise_dist = get_noise_dist(**params)

    if run_single_step:
        w2_bounds = single_step_w2_options(
            dynamics=dynamics,
            noise_dist=noise_dist,
            q=initial_dist,
            w2_p__q_options=[w_p__q],
            run_independent_coupling=False,
            run_lagrangian_duality=True,
            **params
        )
        return w2_bounds

if __name__ == '__main__':
    torch.manual_seed(0)

    num_locs_experiment = [10, 100, 1000, 10000]
    w_p__q = 0.0

    all_dynamics_names = ['Sigmoid (1D)',
                          'Bounded Linear (2D)',
                          'Quadruple-Tank (4D)',
                          'NN Layer (10D)',
                          'Mountain Car (2D)',
                          'Dubins car (3D)'
                         ]
    all_dynamic_types = ['SigmoidDynamics',
                         'BoundedLinearDynamics',
                         'LinearDynamics',
                         'LinearDiagonalSigmoidDynamics',
                         'MountainCarDynamics',
                         'DubinsCarDynamics'
                         ]
    all_num_dims = [1,
                    2,
                    4,
                    10,
                    2,
                    3
                    ]
    all_dyn_settings = [1,
                        0,
                        0,
                        0,
                        0,
                        0
                        ]

    experiment_dict = {}
    for (dynamics_type, num_dims, dyn_setting) in zip(all_dynamic_types, all_num_dims, all_dyn_settings):
        bounds = []
        for num_locs in num_locs_experiment:
            w2_bounds = effect_adding_locs(dynamics_type, num_dims, dyn_setting, num_locs, w_p__q)
            bounds.append(w2_bounds[w_p__q]['lagrangian_duality'].item())

        experiment_dict[dynamics_type] = bounds

    plot.plot_dict(experiment_dict, num_locs_experiment, all_dynamics_names)

