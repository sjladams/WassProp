import torch

from experiments import multi_step
from dynamics import get_dynamics
import plot

from utils import parse_arguments
from utils_distributions import get_initial_dist, get_noise_dist

if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        dynamics_type = "DoubleSpiral2dDynamics",
        dynamics_setting = 0,
        num_locs = 10,
        size_after_compr=10,
        num_samples = 100,
        lr = 0.01,
        num_iterations = 100,
        plot = False
    )

    xlim, ylim = [-1.9, 1.9], [-0.75, 1.25]
    figsize = (14, 8)
    save_by = "double_spiral"
    # save_by = None
    dynamics = get_dynamics(**vars(args))
    plot.plot_2d_dynamics(dynamics, xlim=xlim, ylim=ylim, figsize=figsize, scale=1, save_by=save_by)
    print(f"global lipschitz: {dynamics.global_lipschitz}")
    # raise RuntimeError("DEBUG STOP")
    initial_dist = get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)
    noise_dist = get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    w2_q__sign_q_store, w2_p1__q1_store, samples_store, q_store = multi_step(
        w2_p__q= 0.1,
        w2_noise_dist= 0.01,
        dynamics=dynamics,
        noise_dist=noise_dist,
        q=initial_dist,
        num_time_steps=10,
        run_lagrangian_duality=True,
        run_empirical=False,
        propagate_via_gmm=True,
        num_samples=args.num_samples,
        num_locs=args.num_locs,
        size_after_compr=args.size_after_compr
    )

    plot.plot_2d_ambiguity_balls(samples_store, w2_p1__q1_store, q_store, xlim=xlim, ylim=ylim, figsize=figsize, save_by=save_by)