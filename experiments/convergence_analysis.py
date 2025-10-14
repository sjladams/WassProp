from experiments import single_step_w2_options
from dynamics import get_dynamics
from experiments.utils import parse_arguments
import torch
from utils_distributions import get_initial_dist, get_noise_dist

def convergence_analysis(dynamics_type, dyn_setting, num_locs, w_p__q):
    args = parse_arguments(
        dynamics_type=dynamics_type,
        dynamics_setting=dyn_setting,
        num_locs=num_locs,
        num_locs_after_compr=1,
    )

    dynamics = get_dynamics(**vars(args))
    initial_dist = get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)
    noise_dist = get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    w2_q__disc_q_store, w2_p1__q1_store = single_step_w2_options(
        dynamics=dynamics,
        noise_dist=noise_dist,
        q=initial_dist,
        w2_p__q_options=[w_p__q],
        run_lagrangian_duality=True,
        propagate_via_gmm=True,
        num_samples=args.num_samples,
        num_locs=args.num_locs
    )

    return w2_p1__q1_store

if __name__ == '__main__':
    torch.manual_seed(0)

    # Set parameters
    num_locs_experiment = [10, 100, 1000, 10000]
    w_p__q = 0.0

    run_inputs = { # [dynamics_type, dynamics_setting]
        'Sigmoid (1D)' : ('SigmoidDynamics', 0),
        'Bounded Linear (2D)' : ('BoundedLinearDynamics', 0),
        'Quadruple-Tank (4D)' : ('LinearDynamics', 0),
        'NN Layer (10D)' : ('DiagonalLinearSigmoidDynamics', 2),
        'Mountain Car (2D)' : ('MountainCarDynamics', 0),
        'Dubins car (3D)' : ('DubinsCarDynamics', 0)
    }

    experiment_dict = {}
    for dynamics_name in run_inputs.keys():

        dynamics_type = run_inputs[dynamics_name][0]
        dynamics_setting = run_inputs[dynamics_name][1]

        bounds = []
        for num_locs in num_locs_experiment:
            w2_p1__q1_store = convergence_analysis(dynamics_type, dynamics_setting, num_locs, w_p__q)
            bounds.append(w2_p1__q1_store[w_p__q]['w2_p1__q1_lagrangian_duality'].item())

        experiment_dict[dynamics_name] = bounds

    # Print results (dict)
    print(experiment_dict)