import torch

from experiments import multi_step, single_step_w2_options
from dynamics import get_dynamics
import plot

from utils import parse_arguments
from utils_distributions import get_initial_dist, get_noise_dist
import matplotlib.patches as patches


def patch_creator():
    return [
        patches.Rectangle((-2., -2.), 1., 1., facecolor='grey', edgecolor='black'),
        patches.Rectangle((-1., 0.), 1., 1., facecolor='grey', edgecolor='black'),
        patches.Rectangle((0., -1.), 1., 1., facecolor='grey', edgecolor='black'),
        patches.Rectangle((1., 1.), 1., 1., facecolor='grey', edgecolor='black'),
        patches.Rectangle((-1., -1.), 1., 1., facecolor='white', edgecolor='black'),
        patches.Rectangle((0., 0.), 1., 1., facecolor='white', edgecolor='black')
    ]

def text_creator():
    return [
        dict(x=-1.5, y=-1.55, s='obs', ha='center', va='center', fontsize=32, color='white'),
        dict(x=-0.5, y=0.5, s='obs', ha='center', va='center', fontsize=32, color='white'),
        dict(x=0.5, y=-0.5, s='obs', ha='center', va='center', fontsize=32, color='white'),
        dict(x=1.5, y=1.5, s='obs', ha='center', va='center', fontsize=32, color='white'),
        dict(x=-0.5, y=-0.5, s='tgt', ha='center', va='center', fontsize=32, color='black'),
        dict(x=0.5, y=0.5, s='tgt', ha='center', va='center', fontsize=32, color='black')
    ]


if __name__ == '__main__':
    torch.manual_seed(0)

    configs = {
        0: dict(dynamics_setting= 0, num_time_step=8, save_by="upper_left"),
        1: dict(dynamics_setting=2, num_time_step=10, save_by="lower_right"),
    }

    config = configs[1]

    args = parse_arguments(
        dynamics_type = "SwitchedLinearDynamics",
        dynamics_setting = config['dynamics_setting'],
        num_locs = 100,
        size_after_compr=10,
        num_samples = 100,
        lr = 0.01,
        num_iterations = 100,
        plot = False
    )
    xlim, ylim = [-2., 2.], [-2., 2.]
    figsize = (12, 12)
    save_by = "switched_linear"
    # save_by = None

    dynamics = get_dynamics(**vars(args))
    plot.plot_2d_dynamics(dynamics, patch_creator=patch_creator, text_creator=text_creator,
                          xlim=xlim, ylim=ylim, figsize=figsize, scale=1., save_by=save_by)
    print(f"global lipschitz: {dynamics.global_lipschitz}")
    # raise RuntimeError("DEBUG STOP")
    initial_dist = get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)
    noise_dist = get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    w2_q__sign_q_store, w2_p1__q1_store, samples_store, q_store = multi_step(
        w2_p__q=0.01,
        w2_noise_dist=0.0001,
        dynamics=dynamics,
        noise_dist=noise_dist,
        q=initial_dist,
        num_time_steps=config['num_time_step'],
        run_lagrangian_duality=True,
        run_empirical=False,
        propagate_via_gmm=True,
        num_samples=args.num_samples,
        num_locs=args.num_locs,
        size_after_compr=args.size_after_compr
    )

    plot.plot_2d_ambiguity_balls(samples_store, w2_p1__q1_store, q_store, patch_creator=patch_creator, text_creator=text_creator,
                                 xlim=xlim, ylim=ylim, figsize=figsize, save_by=f"{save_by}_{config['save_by']}")