import torch
import matplotlib.pyplot as plt

from duq_via_wasserstein import multi_step, single_step, AmbiguityBall

from dynamics import get_stoch_dynamics
from handlers import parse_arguments
import utils


def plot(store):
    methods = list(store.keys())
    w2_p__q_options = list(store[methods[0]].keys())

    plt.figure(figsize=(16, 16))
    for key in methods:
        plt.plot(w2_p__q_options, [store[key][w2_p__q] for w2_p__q in w2_p__q_options], label=key)

    plt.legend()
    plt.title(f"{dynamics.state_dynamics.__class__.__name__ if hasattr(dynamics, 'state_dynamics') else dynamics.__class__.__name__} (Lipschitz={dynamics.global_lipschitz:.2f})")
    plt.xlabel('$W_2(p,q)$')
    plt.xticks(w2_p__q_options)
    plt.ylabel(r'$W_2(f p, f \Delta q)$')
    plt.xlim(min(w2_p__q_options), max(w2_p__q_options))
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(0)

    run_single_step = True
    run_multi_step = False

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
                    q=AmbiguityBall(initial_dist, w2_p__q),
                    noise=AmbiguityBall(noise_dist, 0.),
                    num_locs=args.num_locs,
                    use_lagrangian_duality=method == 'lagrangian_duality'
                ).w2

        plot(store)

    if run_multi_step:
        trace = multi_step(
            dynamics=dynamics,
            q=AmbiguityBall(initial_dist, 0.),
            noise = AmbiguityBall(noise_dist, 0.),
            num_time_steps=3,
            num_locs=args.num_locs,
            use_lagrangian_duality= True,
        )
