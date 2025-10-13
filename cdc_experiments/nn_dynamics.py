import torch

from experiments import multi_step
from dynamics import get_dynamics
import plot

from utils import parse_arguments
from utils_distributions import get_initial_dist, get_noise_dist

from quantitative_analysis import hyper_params_analysis, boundary_cond_analysis


def illustrate():
    args = parse_arguments(
        dynamics_type = "NeuralPendulumDynamics",
        dynamics_setting = 0,
        num_locs = 100,
        size_after_compr=10,
        num_samples = 100,
        lr = 0.01,
        num_iterations = 100,
        plot = False
    )

    save_by = "neural_pendulum"
    # save_by = None
    dynamics = get_dynamics(**vars(args))
    plot.plot_2d_dynamics(dynamics, xlim=[-2.0, 1.8], ylim=[-1.8, 2.0], figsize=(12,12), save_by=save_by)
    print(f"global lipschitz: {dynamics.global_lipschitz}")
    initial_dist = get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)
    noise_dist = get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    w2_q__sign_q_store, w2_p1__q1_store, samples_store, q_store = multi_step(
        w2_p__q= 0.001,
        w2_noise_dist= 0.001,
        dynamics=dynamics,
        noise_dist=noise_dist,
        q=initial_dist,
        num_time_steps=20,
        run_lagrangian_duality=True,
        run_empirical=False,
        propagate_via_gmm=True,
        num_samples=args.num_samples,
        num_locs=args.num_locs,
        size_after_compr=args.size_after_compr
    )

    plot.plot_2d_ambiguity_balls(samples_store, w2_p1__q1_store, q_store, xlim=[-2.1, 0.8], ylim=[-1.0, 1.5], figsize=(12,12), save_by=save_by)

if __name__ == '__main__':
    torch.manual_seed(0)

    # illustrate()

    args = parse_arguments(
        dynamics_type = "NeuralPendulumDynamics",
        dynamics_setting = 0,
        num_locs = 100,
        size_after_compr=5,
        num_samples = 100,
        lr = 0.01,
        num_iterations = 100,
        plot = False
    )

    name = "neural_pendulum"

    hyper_params_analysis(args, name, w2_p__q=0.001, w2_noise_dist=0.001)
    boundary_cond_analysis(args, name)

