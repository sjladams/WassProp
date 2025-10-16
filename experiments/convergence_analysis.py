import torch
import pprint

from duq_via_wasserstein import single_step, AmbiguitySet

from handlers import parse_arguments
from dynamics import get_dynamics
import utils


def convergence_analysis(dynamics_type, setting, num_locs, w2_p__q):
    args = parse_arguments(
        dynamics_type=dynamics_type,
        dynamics_setting=setting,
        num_locs=num_locs,
    )

    dynamics = get_dynamics(**vars(args))
    initial_dist = utils.get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)
    noise_dist = utils.get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    results = dict()
    for method in ['global_lipschitz', 'lagrangian_duality']:
        w2 = single_step(
            dynamics=dynamics,
            q=AmbiguitySet(initial_dist, w2_p__q),
            noise=AmbiguitySet(noise_dist, 0.),
            num_locs=args.num_locs,
            use_lagrangian_duality=True
        ).w2
        results[method] = float(w2)
    return results

if __name__ == '__main__':
    torch.manual_seed(0)

    num_locs_experiment = [10, 100, 1000, 10000]
    w2_p__q = 0.0

    run_inputs = { # [dynamics_type, dynamics_setting]
        'Sigmoid (1D)' : ('SigmoidDynamics', 0),
        'Bounded Linear (2D)' : ('BoundedLinearDynamics', 0),
        'Quadruple-Tank (4D)' : ('LinearDynamics', 0),
        'NN Layer (10D)' : ('DiagonalLinearSigmoidDynamics', 2),
        # 'Mountain Car (2D)' : ('MountainCarDynamics', 0),
        # 'Dubins car (3D)' : ('DubinsCarDynamics', 0)
    }

    results = {}
    for dynamics_name, (dynamics_type, setting) in run_inputs.items():
        results[dynamics_name] = {}
        for num_locs in num_locs_experiment:
            results[dynamics_name][num_locs] = convergence_analysis(dynamics_type, setting, num_locs, w2_p__q)

    pprint.pprint(results)