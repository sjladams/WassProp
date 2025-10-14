import torch

from propagation import multi_step, multi_step_empirical, SampledPath
from dynamics import get_dynamics
from utils_distributions import AmbiguitySet

import experiments.plot as plot
from experiments.handlers import parse_arguments
import experiments.utils as utils


def multistep_approximation(dynamics_type, setting, num_locs):
    args = parse_arguments(
        dynamics_type=dynamics_type,
        dynamics_setting=setting,
        num_locs=num_locs,
        num_samples=500
    )

    num_time_steps = 10

    dynamics = get_dynamics(**vars(args))
    initial_dist = utils.get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)
    noise_dist = utils.get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    q = AmbiguitySet(initial_dist, 0.1)
    noise = AmbiguitySet(noise_dist, 0.0)

    path = multi_step(
        dynamics=dynamics, 
        q=q, 
        noise=noise,
        num_time_steps=num_time_steps,
        use_lagrangian_duality=True,
        num_locs=args.num_locs,
    )

    true_samples = multi_step_empirical(
        dynamics=dynamics,
        p_emp=q.sample(args.num_samples),
        noise=noise,
        num_time_steps=num_time_steps,
        num_samples=args.num_samples,
    )
    approx_samples = SampledPath({k: path.at(k).sample(args.num_samples) for k in path.ordered_indices})

    plot.plot_multi_step(dynamics=dynamics, true_samples=true_samples, approx_samples=approx_samples)


if __name__ == '__main__':
    torch.manual_seed(0)

    dynamics_type = 'MountainCarDynamics'
    num_locs = 100
    dyn_setting = 0

    multistep_approximation(dynamics_type, dyn_setting, num_locs)