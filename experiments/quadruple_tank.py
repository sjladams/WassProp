import torch

from duq_via_wasserstein import multi_step, AmbiguityBall

from dynamics import get_stoch_dynamics
from handlers import parse_arguments
import utils
from analysis import hyper_params_analysis, boundary_cond_analysis


def illustrate():
    args = parse_arguments(
        dynamics_type = "LinearDynamics",
        dynamics_setting = 1,
        num_locs = 10,
        num_samples = 100,
        save = False
    )

    dynamics = get_stoch_dynamics(name=args.dynamics_type, **vars(args.dynamics))
    initial_dist = utils.get_initial_dist(loc=args.initial_dist.loc, variance=args.initial_dist.variance)
    noise_dist = utils.get_noise_dist(loc=args.noise_dist.loc, variance=args.noise_dist.variance)

    q = AmbiguityBall(initial_dist, 0.1)
    noise = AmbiguityBall(noise_dist, 0.01)

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

    hyper_params_analysis(args, name)
    boundary_cond_analysis(args, name)


if __name__ == '__main__':
    torch.manual_seed(0)

    illustrate()

    quantitative_analysis()

    




