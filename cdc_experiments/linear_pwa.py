import torch

from experiments import multi_step, single_step_w2_options
from dynamics import get_dynamics
import plot

from utils import parse_arguments
from utils_distributions import get_initial_dist, get_noise_dist

if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        dynamics_type = "PiecewiseAffine4modes2dDynamics",
        dynamics_setting = 0,
        num_locs = 10,
        size_after_compr=10,
        num_samples = 100,
        lr = 0.01,
        num_iterations = 100,
        plot = False
    )

    dynamics = get_dynamics(**vars(args))
    initial_dist = get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)
    noise_dist = get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    plot.plot_2d_dynamics(dynamics)

    w2_q__sign_q_store, w2_p1__q1_store, samples_store, q_store = multi_step(
        dynamics=dynamics,
        noise_dist=noise_dist,
        q=initial_dist,
        num_time_steps=20,
        run_lagrangian_duality=True,
        run_empirical=False,
        propagate_via_gmm=False,
        num_samples=args.num_samples,
        num_locs=args.num_locs,
        size_after_compr=args.size_after_compr
    )

    plot.plot_multi_step(dynamics, samples_store, layout="scatter")