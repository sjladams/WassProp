import torch

from duq_via_wasserstein import multi_step, get_dynamics, AmbiguitySet

from handlers import parse_arguments
import utils


def illustrate():
    args = parse_arguments(
        dynamics_type = "LinearDynamics",
        dynamics_setting = 1,
        num_locs = 10,
        num_samples = 100,
        save = False
    )

    dynamics = get_dynamics(**vars(args))
    print(f"global lipschitz: {dynamics.global_lipschitz}")
    initial_dist = utils.get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)

    noise_dist = utils.get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    q = AmbiguitySet(initial_dist, 0.1)
    noise = AmbiguitySet(noise_dist, 0.01)

    path = multi_step(
        dynamics=dynamics, 
        q=q, 
        noise=noise,
        num_time_steps=10,
        use_lagrangian_duality=True,
        num_locs=args.num_locs,
    )

def quantitative_analysis():
    args = parse_arguments(
            dynamics_type="LinearDynamics",
            dynamics_setting=1,
            num_locs=10,
            num_samples=100,
            save=False
        )
    
    name = "quadruple_tank"

    utils.hyper_params_analysis(args, name)
    utils.boundary_cond_analysis(args, name)


if __name__ == '__main__':
    torch.manual_seed(0)

    illustrate()

    quantitative_analysis()

    




