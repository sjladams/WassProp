import torch

from duq_via_wasserstein import multi_step, multi_step_empirical, SampledPath, AmbiguityBall

from handlers import parse_arguments
from dynamics import get_stoch_dynamics
import plot
import utils


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        dynamics_type = "Spiral2dDynamics",
        dynamics_setting = 0,
        num_locs = 10,
        num_samples = 100,
        save = False
    )
    
    num_time_steps = 20

    save_by = f"{args.results_folder}spiral"
    xlim, ylim = [-1., 1.], [-1., 1.]
    dynamics = get_stoch_dynamics(name=args.dynamics_type, **vars(args.dynamics))
    plot.plot_2d_dynamics(
        dynamics, 
        xlim=xlim, ylim=ylim, 
        save_by=f"{save_by}_dynamics", save=args.save
    )
    print(f"global lipschitz: {dynamics.global_lipschitz}")

    initial_dist = utils.get_initial_dist(loc=args.initial_dist.loc, variance=args.initial_dist.variance)
    noise_dist = utils.get_noise_dist(loc=args.noise_dist.loc, variance=args.noise_dist.variance)

    q = AmbiguityBall(initial_dist, 0.1)
    noise = AmbiguityBall(noise_dist, 0.01)

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
    )

    approx_samples = SampledPath({k: path.at(k).sample(args.num_samples) for k in path.ordered_indices})

    plot.plot_2d_ambiguity_balls(
        true_samples, path, 
        xlim=xlim, ylim=ylim,
        save_by=f"{save_by}_path_true", save=args.save
    )
    plot.plot_2d_ambiguity_balls(
        approx_samples, path, 
        xlim=xlim, ylim=ylim,
        save_by=f"{save_by}_path_appr", save=args.save
    )