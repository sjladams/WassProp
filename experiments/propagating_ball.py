import torch
import pprint

from duq_via_wasserstein import single_step, AmbiguitySet

from dynamics import get_stoch_dynamics
from handlers import parse_arguments
import utils


def propagate_wasserstein_ball(dynamics_type, setting, num_locs, w2_p__q):
    args = parse_arguments(
        dynamics_type=dynamics_type,
        dynamics_setting=setting,
        num_locs=num_locs,
    )

    dynamics = get_stoch_dynamics(name=args.dynamics_type, **vars(args.dynamics))
    initial_dist = utils.get_initial_dist(loc=args.initial_dist.loc, variance=args.initial_dist.variance)
    noise_dist = utils.get_noise_dist(loc=args.noise_dist.loc, variance=args.noise_dist.variance)

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

    w2_p__q_options = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    run_inputs = { # (dynamics_type, dynamics_setting, num_locs)
        'Sigmoid (1D)' : ('SigmoidDynamics', 0, 100),
        'Bounded Linear (2D)' : ('BoundedLinearDynamics', 0, 100),
        'Quadruple-Tank (4D)' : ('LinearDynamics', 0, 1000),
        'NN Layer (10D)' : ('DiagonalSigmoidDynamics', 2, 1000),
        # 'Mountain Car (2D)' : ('MountainCarDynamics', 0, 100),
        # 'Dubins car (3D)' : ('DubinsCarDynamics', 0, 1000)
    }

    results = {}
    for dynamics_name, (dynamics_type, setting, num_locs) in run_inputs.items():
        results[dynamics_name] = {}
        for w2_p__q in w2_p__q_options:
            results[dynamics_name][w2_p__q] = propagate_wasserstein_ball(dynamics_type, setting, num_locs, w2_p__q)

    pprint.pprint(results)
