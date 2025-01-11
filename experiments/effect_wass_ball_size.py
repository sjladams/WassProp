import torch

from experiments import multi_step, get_noise_dist, get_initial_dist, single_step_w2_options
from dynamics import get_dynamics
import plot

import os
import json

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

    w_p__q = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    run_inputs = { # [dynamics_type, num_dims, dynamics_setting, num_locs]
        'Sigmoid (1D)' : ('SigmoidDynamics', 1, 1, 100),
        'Bounded Linear (2D)' : ('BoundedLinearDynamics', 2, 0, 100),
        'Quadruple-Tank (4D)' : ('LinearDynamics', 4, 0, 100),
        #'NN Layer (10D)' : ('LinearDiagonalSigmoidDynamics', 10, 0, 100),
        'Mountain Car (2D)' : ('MountainCarDynamics', 2, 0, 100),
        'Dubins car (3D)' : ('DubinsCarDynamics', 3, 0, 100)
    }

    experiment_duality_dict, experiment_lipschitz_dict = {}, {}
    for dynamics_name in run_inputs.keys():

        dynamics_type = run_inputs[dynamics_name][0]
        num_dims = run_inputs[dynamics_name][1]
        dynamics_setting = run_inputs[dynamics_name][2]
        num_locs = run_inputs[dynamics_name][3]

        w2_bounds = effect_wass_ball_size(dynamics_type, num_dims, dynamics_setting, num_locs, w_p__q)

        bounds_duality, bounds_lipschitz  = [], []
        for w in w_p__q:
            bounds_duality.append(w2_bounds[w]['lagrangian_duality'].item())
            bounds_lipschitz.append(w2_bounds[w]['global_lipschitz'].item())

        experiment_duality_dict[dynamics_type] = bounds_duality
        experiment_lipschitz_dict[dynamics_type] = bounds_lipschitz


    folder = r"C:\Users\efigueiredomot\Desktop\Papers\Wasserstein"
    os.makedirs(folder, exist_ok=True)  # Create the folder if it doesn't exist

    file_path = os.path.join(folder, f"wass_ball_lipschitz.json")
    with open(file_path, "w") as file:
        json.dump(experiment_lipschitz_dict, file)

    file_path = os.path.join(folder, f"wass_ball_lagrangian.json")
    with open(file_path, "w") as file:
        json.dump(experiment_duality_dict, file)