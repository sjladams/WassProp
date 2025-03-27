import torch
from experiments import multi_step
from dynamics import get_dynamics

from utils import parse_arguments
from utils_distributions import get_initial_dist, get_noise_dist
from quantitative_analysis import hyper_params_analysis, boundary_cond_analysis


def test():
    args = parse_arguments(
        dynamics_type = "LinearDynamics",
        dynamics_setting = 1,
        num_locs = 10,
        size_after_compr=10,
        num_samples = 100,
        lr = 0.01,
        num_iterations = 100,
        plot = False
    )

    dynamics = get_dynamics(**vars(args))
    print(f"global lipschitz: {dynamics.global_lipschitz}")
    initial_dist = get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)

    noise_dist = get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    w2_q__sign_q_store, w2_p1__q1_store, samples_store, q_store = multi_step(
        w2_p__q= 0.1,
        w2_noise_dist= 0.1,
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


if __name__ == '__main__':
    torch.manual_seed(0)

    # test()

    args = parse_arguments(
        dynamics_type="LinearDynamics",
        dynamics_setting=1,
        num_locs=10,
        size_after_compr=10,
        num_samples=100,
        lr=0.01,
        num_iterations=100,
        plot=False
    )
    name = "quadruple_tank"

    hyper_params_analysis(args, name)
    boundary_cond_analysis(args, name)




