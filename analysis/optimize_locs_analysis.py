from experiments import multi_step, get_noise_dist, get_initial_dist, single_step_w2_options
from dynamics import get_dynamics
from utils import load_params, parse_arguments
import torch
import json
import os
from configs import FOLDER, W_P__Q_CHOICES_OPTIMIZE_LOCS

def locs_optimization_analysis(dynamics_type, num_dims, dyn_setting, num_locs, w_p__q, optimize):
    args = parse_arguments(
        dynamics_type=dynamics_type,
        num_dims=num_dims,
        dynamics_setting=dyn_setting,
        num_locs=num_locs,
        num_locs_after_compr=1,
        num_samples=1000,
        lr=0.01,
        num_iterations=10,
        plot=False,
        optimize_locs=optimize
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

    run_inputs = { # [dynamics_type, num_dims, dynamics_setting, num_locs]
        #'Sigmoid (1D)' : ('SigmoidDynamics', 1, 1, 100),
        #'Bounded Linear (2D)' : ('BoundedLinearDynamics', 2, 0, 100),
        #'Quadruple-Tank (4D)' : ('LinearDynamics', 4, 0, 100),
        #'NN Layer (10D)' : ('LinearDiagonalSigmoidDynamics', 10, 0, 100),
        'Mountain Car (2D)' : ('MountainCarDynamics', 2, 0, 100),
        #'Dubins car (3D)' : ('DubinsCarDynamics', 3, 0, 100)
    }

    w_p__q = W_P__Q_CHOICES_OPTIMIZE_LOCS

    experiment_duality_dict, experiment_lipschitz_dict, experiment_duality_optimize_dict = {}, {}, {}
    for dynamics_name in run_inputs.keys():

        dynamics_type = run_inputs[dynamics_name][0]
        num_dims = run_inputs[dynamics_name][1]
        dynamics_setting = run_inputs[dynamics_name][2]
        num_locs = run_inputs[dynamics_name][3]

        w2_bounds = locs_optimization_analysis(dynamics_type, num_dims, dynamics_setting, num_locs, w_p__q, False)
        w2_bounds_optimize = locs_optimization_analysis(dynamics_type, num_dims, dynamics_setting, num_locs, w_p__q, True)

        bounds_duality, bounds_lipschitz, bounds_duality_optimize = [], [], []
        for w in w_p__q:
            bounds_duality.append(w2_bounds[w]['lagrangian_duality'].item())
            bounds_lipschitz.append(w2_bounds[w]['global_lipschitz'].item())
            bounds_duality_optimize.append(w2_bounds_optimize[w]['lagrangian_duality'].item())

        experiment_lipschitz_dict[dynamics_type] = bounds_lipschitz
        experiment_duality_dict[dynamics_type] = bounds_duality
        experiment_duality_optimize_dict[dynamics_type] = bounds_duality_optimize

        # Create dictionary with results
        folder = FOLDER
        os.makedirs(folder, exist_ok=True)  # Create the folder if it doesn't exist

        file_path = os.path.join(folder, f"optimize_locs_lipschitz.json")
        with open(file_path, "w") as file:
            json.dump(experiment_lipschitz_dict, file)

        file_path = os.path.join(folder, f"optimize_locs_duality_false.json")
        with open(file_path, "w") as file:
            json.dump(experiment_duality_dict, file)

        file_path = os.path.join(folder, f"optimize_locs_duality_true.json")
        with open(file_path, "w") as file:
            json.dump(experiment_duality_optimize_dict, file)