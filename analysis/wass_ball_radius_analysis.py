from experiments import multi_step, get_noise_dist, get_initial_dist, single_step_w2_options
from dynamics import get_dynamics
from utils import load_params, parse_arguments
import torch
import json
import os
from configs import FOLDER, W_P__Q_CHOICES

def wass_ball_radius_analysis(dynamics_type, num_dims, dyn_setting, num_locs, w_p__q):
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

    w_p__q = W_P__Q_CHOICES

    run_inputs = { # [dynamics_type, num_dims, dynamics_setting, num_locs]
        'Sigmoid (1D)' : ('SigmoidDynamics', 1, 1, 100),
        'Bounded Linear (2D)' : ('BoundedLinearDynamics', 2, 0, 100),
        'Quadruple-Tank (4D)' : ('LinearDynamics', 4, 0, 1000),
        'NN Layer (10D)' : ('LinearDiagonalSigmoidDynamics', 10, 0, 1000),
        'Mountain Car (2D)' : ('MountainCarDynamics', 2, 0, 100),
        'Dubins car (3D)' : ('DubinsCarDynamics', 3, 0, 1000)
    }

    experiment_duality_dict, experiment_lipschitz_dict, experiment_diff_dict = {}, {}, {}
    for dynamics_name in run_inputs.keys():

        dynamics_type = run_inputs[dynamics_name][0]
        num_dims = run_inputs[dynamics_name][1]
        dynamics_setting = run_inputs[dynamics_name][2]
        num_locs = run_inputs[dynamics_name][3]

        w2_bounds = wass_ball_radius_analysis(dynamics_type, num_dims, dynamics_setting, num_locs, w_p__q)

        bounds_duality, bounds_lipschitz  = [], []
        for w in w_p__q:
            bounds_duality.append(w2_bounds[w]['lagrangian_duality'].item())
            bounds_lipschitz.append(w2_bounds[w]['global_lipschitz'].item())

        experiment_duality_dict[dynamics_name] = bounds_duality
        experiment_lipschitz_dict[dynamics_name] = bounds_lipschitz
        experiment_diff_dict[dynamics_name] = [a - b for a, b in zip(bounds_lipschitz, bounds_duality)]


    # Create dictionary with results
    folder = FOLDER
    os.makedirs(folder, exist_ok=True)  # Create the folder if it doesn't exist

    file_path = os.path.join(folder, f"wass_ball_radius_analysis_lipschitz.json")
    with open(file_path, "w") as file:
        json.dump(experiment_lipschitz_dict, file)

    file_path = os.path.join(folder, f"wass_ball_radius_analysis_lagrangian.json")
    with open(file_path, "w") as file:
        json.dump(experiment_duality_dict, file)

    file_path = os.path.join(folder, f"wass_ball_radius_analysis_difference.json")
    with open(file_path, "w") as file:
        json.dump(experiment_diff_dict, file)