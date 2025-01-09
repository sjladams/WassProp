import torch

from experiments import multi_step, get_noise_dist, get_initial_dist, single_step_w2_options
from dynamics import get_dynamics
import plot

from utils import load_params, parse_arguments

def effect_wass_ball_size(dynamics_type, num_dims, dyn_setting, num_locs, w_p__q):
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
            w2_p__q_options=w_p__q,
            run_independent_coupling=False,
            run_lagrangian_duality=True,
            **params
        )
        return w2_bounds

if __name__ == '__main__':
    torch.manual_seed(0)

    w_p__q = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    all_dynamics_names = ['Sigmoid (1D)',
                          'Bounded Linear (2D)',
                          'Linear (5D)',
                          'Toy NN Layer (10D)',
                          'Mountain Car (2D)'
                         ]
    all_dynamic_types = ['SigmoidDynamics',
                         'BoundedLinearDynamics',
                         'LinearDynamics',
                         'LinearDiagonalSigmoidDynamics',
                         'MountainCarDynamics',
                         ]
    all_num_dims = [1,
                    2,
                    5,
                    10,
                    2
                    ]

    all_dyn_settings = [0,
                        0,
                        0,
                        0,
                        0
                        ]

    all_num_locs = [100,
                    100,
                    100,
                    100,
                    100
                    ]

    experiment_duality_dict, experiment_lipschitz_dict, diff_dict = {}, {}, {}
    for (dynamics_type, num_dims, dyn_setting, num_locs) in zip(all_dynamic_types, all_num_dims, all_dyn_settings, all_num_locs):
        w2_bounds = effect_wass_ball_size(dynamics_type, num_dims, dyn_setting, num_locs, w_p__q)

        bounds_duality, bounds_lipschitz,  = [], []
        for w in w_p__q:
            bounds_duality.append(w2_bounds[w]['lagrangian_duality'].item())
            bounds_lipschitz.append(w2_bounds[w]['global_lipschitz'].item())

        diff = [a - b for a, b in zip(bounds_lipschitz, bounds_duality)]

        experiment_duality_dict[dynamics_type] = bounds_duality
        experiment_lipschitz_dict[dynamics_type] = bounds_lipschitz
        diff_dict[dynamics_type] = diff

    plot.plot_dict(experiment_duality_dict,
                   w_p__q,
                   all_dynamics_names,
                   x_label=r"$\theta$",
                   y_label= r"$\sup_{ \mathbb{Q} \in \mathbb{B}_{\theta}(\mathbb{P})}  \mathbb{W}_{\rho}(f\#\mathbb{Q}, f\#\Delta_{\mathcal{R}, \mathcal{C}}\#\mathbb{P})$",
                   log_scale=False)

    plot.plot_dict(diff_dict,
                   w_p__q,
                   all_dynamics_names,
                   x_label=r"$\theta$",
                   y_label="Difference between the global Lipschitz bounds and ours",
                   log_scale=False)


