from experiments import single_step_w2_options
from dynamics import get_dynamics
from utils import load_params, parse_arguments
import torch
from utils_distributions import get_initial_dist, get_noise_dist

def convergence_analysis(dynamics_type, num_dims, dyn_setting, num_locs, w_p__q):
    args = parse_arguments(
        dynamics_type=dynamics_type,
        num_dims=num_dims,
        dynamics_setting=dyn_setting,
        num_locs=num_locs,
        num_locs_after_compr=1,
    )
    params = load_params(args)

    dynamics = get_dynamics(**params)
    initial_dist = get_initial_dist(**params)
    noise_dist = get_noise_dist(**params)

    w2_bounds = single_step_w2_options(
        dynamics=dynamics,
        noise_dist=noise_dist,
        q=initial_dist,
        w2_p__q_options=[w_p__q],
        run_independent_coupling=False,
        run_lagrangian_duality=True,
        propagate_via_gmm=True,
        **params
    )

    return w2_bounds

if __name__ == '__main__':
    torch.manual_seed(0)

    # Set parameters
    num_locs_experiment = [10]
    w_p__q = 0.0

    run_inputs = { # [dynamics_type, num_dims, dynamics_setting]
        'Sigmoid (1D)' : ('SigmoidDynamics', 1, 0),
        'Bounded Linear (2D)' : ('BoundedLinearDynamics', 2, 0),
        'Quadruple-Tank (4D)' : ('LinearDynamics', 4, 0),
        'NN Layer (10D)' : ('LinearDiagonalSigmoidDynamics', 10, 0),
        'Mountain Car (2D)' : ('MountainCarDynamics', 2, 0),
        'Dubins car (3D)' : ('DubinsCarDynamics', 3, 0)
    }

    experiment_dict = {}
    for dynamics_name in run_inputs.keys():

        dynamics_type = run_inputs[dynamics_name][0]
        num_dims = run_inputs[dynamics_name][1]
        dynamics_setting = run_inputs[dynamics_name][2]

        bounds = []
        for num_locs in num_locs_experiment:
            w2_bounds = convergence_analysis(dynamics_type, num_dims, dynamics_setting, num_locs, w_p__q)
            bounds.append(w2_bounds[w_p__q]['lagrangian_duality'].item())

        experiment_dict[dynamics_name] = bounds

    print(experiment_dict)