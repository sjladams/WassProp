import torch

from propagation import multi_step, multi_step_empirical, SampledPath
from dynamics import get_dynamics
from utils_distributions import AmbiguitySet

import experiments.plot as plot
from experiments.handlers import parse_arguments
import experiments.utils as utils


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        dynamics_type = "DoubleSpiral2dDynamics",
        dynamics_setting = 0,
        num_locs = 10,
        num_samples = 100,
        save = False
    )

    num_time_steps = 10

    save_by = f"{args.results_folder}double_spiral"
    dynamics = get_dynamics(**vars(args))
    plot.plot_2d_dynamics(
        dynamics, 
        xlim= [-2.0, 2.0], ylim=[-1.0, 1.0], figsize=(13, 8), scale=None, 
        save_by=f"{save_by}_dynamics", save=args.save
    )
    print(f"global lipschitz: {dynamics.global_lipschitz}")

    initial_dist = utils.get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)
    noise_dist = utils.get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    q = AmbiguitySet(initial_dist, 0.1)
    noise = AmbiguitySet(noise_dist, 0.01)

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

    plot.plot_2d_ambiguity_balls(
        true_samples, path, 
        xlim=[-1.0, 1.0], ylim=[-1., 1.0], figsize=(9, 8), 
        save_by=f"{save_by}_path_true", save=args.save
    )
    plot.plot_2d_ambiguity_balls(
        approx_samples, path, 
        xlim=[-1.0, 1.0], ylim=[-1., 1.0], figsize=(9, 8), 
        save_by=f"{save_by}_path_appr", save=args.save
    )
    





