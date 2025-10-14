import torch

from propagation import multi_step, single_step
from dynamics import get_dynamics
from utils_distributions import AmbiguitySet

from experiments.handlers import parse_arguments
import experiments.plot as plot
from experiments.utils import get_initial_dist, get_noise_dist


if __name__ == '__main__':
    torch.manual_seed(0)

    run_single_step = True
    run_multi_step = False

    args = parse_arguments(
        dynamics_type = "SigmoidDynamics",
        dynamics_setting = 0,
        num_locs = 10,
        size_after_compr=1,
        num_samples = 1000,
        lr = 0.01,
        num_iterations = 1000,
        plot = False
    )

    dynamics = get_dynamics(**vars(args))
    initial_dist = get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)
    noise_dist = get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    if run_single_step:
        store = dict()
        for method in ['global_lipschitz', 'lagrangian_duality']:
            store[method] = dict()
            for w2_p__q in [0., 0.1, 0.5, 1.0, 1.5, 2.0]:
                print(f"\n ------ W_2(p,q) = {w2_p__q} ------ \n")
                store[method][w2_p__q] = single_step(
                    dynamics=dynamics,
                    q=AmbiguitySet(initial_dist, w2_p__q),
                    noise=AmbiguitySet(noise_dist, 0.),
                    num_locs=args.num_locs,
                    use_lagrangian_duality=True
                ).w2

        plot.plot_single_step(dynamics, store)

    if run_multi_step:
        trace = multi_step(
            dynamics=dynamics,
            q=AmbiguitySet(initial_dist, 0.),
            noise = AmbiguitySet(noise_dist, 0.),
            num_time_steps=3,
            num_locs=args.num_locs,
            use_lagrangian_duality= True,
        )
