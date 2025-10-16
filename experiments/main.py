import torch

from duq_via_wasserstein import multi_step, single_step, AmbiguitySet

from dynamics import get_stoch_dynamics
from handlers import parse_arguments
import plot
import utils


if __name__ == '__main__':
    torch.manual_seed(0)

    run_single_step = True
    run_multi_step = True

    args = parse_arguments(
        dynamics_type = "SigmoidDynamics",
        dynamics_setting = 0,
        num_locs = 10,
        num_samples = 1000,
        save=False
    )

    dynamics = get_stoch_dynamics(name=args.dynamics_type, **vars(args.dynamics))
    initial_dist = utils.get_initial_dist(loc=args.initial_dist.loc, variance=args.initial_dist.variance)
    noise_dist = utils.get_noise_dist(loc=args.noise_dist.loc, variance=args.noise_dist.variance)

    if run_single_step:
        store = dict()
        for method in ['global_lipschitz', 'lagrangian_duality']:
            store[method] = dict()
            for w2_p__q in [0., 0.1, 0.5, 1.0, 1.5, 2.0]:
                print(f" ------ W_2(p,q) = {w2_p__q} ------")
                store[method][w2_p__q] = single_step(
                    dynamics=dynamics,
                    q=AmbiguitySet(initial_dist, w2_p__q),
                    noise=AmbiguitySet(noise_dist, 0.),
                    num_locs=args.num_locs,
                    use_lagrangian_duality=method == 'lagrangian_duality'
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
