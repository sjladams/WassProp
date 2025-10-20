import torch
import matplotlib.patches as patches
import argparse

from duq_via_wasserstein import multi_step, multi_step_empirical, SampledPath, AmbiguityBall

from handlers import parse_arguments, param_handler
from dynamics import get_stoch_dynamics
import plot
import utils



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


def run(args):
    dynamics = get_stoch_dynamics(name=args.dynamics_type, **vars(args.dynamics))
    plot.plot_2d_dynamics(
        dynamics, 
        patch_creator=patch_creator, text_creator=text_creator, xlim=[-2., 2.], ylim=[-2., 2.], figsize=(12, 12), scale=None, 
        save_by=f"{args.results_folder}dynamics_{args.tag}", save=args.save
    )
    print(f"global lipschitz: {dynamics.global_lipschitz}")
    initial_dist = utils.get_initial_dist(loc=args.initial_dist.loc, variance=args.initial_dist.variance)
    noise_dist = utils.get_noise_dist(loc=args.noise_dist.loc, variance=args.noise_dist.variance)

    q = AmbiguityBall(initial_dist, 0.01)
    noise = AmbiguityBall(noise_dist, 0.0001)

    path = multi_step(
        dynamics=dynamics, 
        q=q, 
        noise=noise,
        num_time_steps=args.num_time_steps,
        use_lagrangian_duality=True,
        num_locs=args.num_locs,
    )

    true_samples = multi_step_empirical(
        dynamics=dynamics,
        p_emp=q.sample(args.num_samples),
        noise=noise,
        num_time_steps=args.num_time_steps,
        num_samples=args.num_samples,
    )
    approx_samples = SampledPath({k: path.at(k).sample(args.num_samples) for k in path.ordered_indices})

    return path, true_samples, approx_samples



if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        dynamics_type="SwitchedLinearDynamics",
        num_locs=100,
        num_samples=100,
        save=False
    )
    args.name_dynamics = "switched_linear"

    configs = [
        argparse.Namespace(dynamics_setting= 0, num_time_step=8, tag="upper_left"),
        # argparse.Namespace(dynamics_setting=2, num_time_step=10, tag="lower_right"),
    ]

    store = dict()
    for config in configs:
        args.dynamics_setting = config.dynamics_setting
        args.num_time_steps = config.num_time_step
        args.tag = config.tag
        args = param_handler(args)
        
        store[args.tag] = run(args)

    save_by = f"{args.results_folder}switched_linear"

    for tag, (path, true_samples, approx_samples) in store.items():
        plot.plot_2d_ambiguity_balls(
            true_samples, path,
            patch_creator=patch_creator, text_creator=text_creator, xlim=[-2., 2.], ylim=[-2., 2.], figsize=(12, 12), 
            save_by=f"{save_by}_{tag}_path_true", save=args.save
        )

        plot.plot_2d_ambiguity_balls(
            approx_samples, path,
            patch_creator=patch_creator, text_creator=text_creator, xlim=[-2., 2.], ylim=[-2., 2.], figsize=(12, 12), 
            save_by=f"{save_by}_{tag}_path_approx", save=args.save
        )

