from typing import List, Optional, Tuple, Callable
import torch
import matplotlib.patches as patches

from duq_via_wasserstein import multi_step, multi_step_empirical, SampledPath, AmbiguityBall
from dataclasses import dataclass, field

from handlers import parse_arguments
import plot
import utils
from dynamics import get_stoch_dynamics
from analysis import hyper_params_analysis, boundary_cond_analysis


def patch_creator_switched_linear():
    return [
        patches.Rectangle((-2., -2.), 1., 1., facecolor='grey', edgecolor='black'),
        patches.Rectangle((-1., 0.), 1., 1., facecolor='grey', edgecolor='black'),
        patches.Rectangle((0., -1.), 1., 1., facecolor='grey', edgecolor='black'),
        patches.Rectangle((1., 1.), 1., 1., facecolor='grey', edgecolor='black'),
        patches.Rectangle((-1., -1.), 1., 1., facecolor='white', edgecolor='black'),
        patches.Rectangle((0., 0.), 1., 1., facecolor='white', edgecolor='black')
    ]

def text_creator_switched_linear():
    return [
        dict(x=-1.5, y=-1.55, s='obs', ha='center', va='center', fontsize=32, color='white'),
        dict(x=-0.5, y=0.5, s='obs', ha='center', va='center', fontsize=32, color='white'),
        dict(x=0.5, y=-0.5, s='obs', ha='center', va='center', fontsize=32, color='white'),
        dict(x=1.5, y=1.5, s='obs', ha='center', va='center', fontsize=32, color='white'),
        dict(x=-0.5, y=-0.5, s='tgt', ha='center', va='center', fontsize=32, color='black'),
        dict(x=0.5, y=0.5, s='tgt', ha='center', va='center', fontsize=32, color='black')
    ]

@dataclass
class PlotConfig:
    xlim: Optional[List[float]] = None
    ylim: Optional[List[float]] = None
    figsize: Optional[Tuple[int, int]] = None
    patch_creator: Optional[Callable[[], List[patches.Patch]]] = None
    text_creator: Optional[Callable[[], List[dict]]] = None

@dataclass
class RadiiConfig:
    initial: float
    noise: float

@dataclass
class ExpConfig:
    dynamics_type: str
    dynamics_setting: int
    num_locs: int
    num_samples: int
    num_time_steps: int
    radii: RadiiConfig
    dynamics_plot: PlotConfig = field(default_factory=PlotConfig)
    path_plot: PlotConfig = field(default_factory=PlotConfig)

paper_configs = dict(
    spiral=ExpConfig(
        dynamics_type='Spiral2dDynamics',
        dynamics_setting=0,
        num_locs=10,
        num_samples=100,
        num_time_steps=20,
        radii=RadiiConfig(initial=0.1, noise=0.01),
        dynamics_plot=PlotConfig(xlim=[-1., 1.], ylim=[-1., 1.]),
        path_plot=PlotConfig(xlim=[-1., 1.], ylim=[-1., 1.]),
    ),
    double_spiral = ExpConfig(
        dynamics_type='DoubleSpiral2dDynamics',
        dynamics_setting=0,
        num_locs=10,
        num_samples=100,
        num_time_steps=10,
        radii=RadiiConfig(initial=0.1, noise=0.01),
        dynamics_plot=PlotConfig(xlim=[-2., 2.], ylim=[-1., 1.], figsize=(13, 8)),
        path_plot=PlotConfig(xlim=[-1., 1.], ylim=[-1., 1.], figsize=(9, 8)),
    ),
    switched_linear_upper_left = ExpConfig(
        dynamics_type='SwitchedLinearDynamics',
        dynamics_setting=0,
        num_locs=100,
        num_samples=100,
        num_time_steps=8,
        radii=RadiiConfig(initial=0.01, noise=0.0001),
        dynamics_plot=PlotConfig(xlim=[-2., 2.], ylim=[-2., 2.], figsize=(12, 12), patch_creator=patch_creator_switched_linear, text_creator=text_creator_switched_linear),
        path_plot=PlotConfig(xlim=[-2., 2.], ylim=[-2., 2.], figsize=(12, 12), patch_creator=patch_creator_switched_linear, text_creator=text_creator_switched_linear), 
    ),
    switched_linear_lower_right = ExpConfig(
        dynamics_type='SwitchedLinearDynamics',
        dynamics_setting=2,
        num_locs=100,
        num_samples=100,
        num_time_steps=8,
        radii=RadiiConfig(initial=0.01, noise=0.0001),
        dynamics_plot=PlotConfig(xlim=[-2., 2.], ylim=[-2., 2.], figsize=(12, 12), patch_creator=patch_creator_switched_linear, text_creator=text_creator_switched_linear),
        path_plot=PlotConfig(xlim=[-2., 2.], ylim=[-2., 2.], figsize=(12, 12), patch_creator=patch_creator_switched_linear, text_creator=text_creator_switched_linear), 
    ),
    neural_pendulum = ExpConfig(
        dynamics_type='NeuralPendulumDynamics',
        dynamics_setting=0,
        num_locs=100,
        num_samples=100,
        num_time_steps=20,
        radii=RadiiConfig(initial=0.001, noise=0.001),
        dynamics_plot=PlotConfig(xlim=[-2., 1.8], ylim=[-1.8, 2.0], figsize=(12, 12)),
        path_plot=PlotConfig(xlim=[-2.1, 0.8], ylim=[-1., 1.5], figsize=(12, 12)),
    ),
    quadruple_tank = ExpConfig(
        dynamics_type='LinearDynamics',
        dynamics_setting=1,
        num_locs=10,
        num_samples=100,
        num_time_steps=10,
        radii=RadiiConfig(initial=0.1, noise=0.01),
    )
)

def illustrate(name: str):
    config = paper_configs[name]

    args = parse_arguments(
        dynamics_type = config.dynamics_type,
        dynamics_setting = config.dynamics_setting,
        num_locs = config.num_locs,
        num_samples = config.num_samples,
        save = False
    )

    num_time_steps = config.num_time_steps

    save_by = f"{args.results_folder}{name}"
    dynamics = get_stoch_dynamics(name=args.dynamics_type, **vars(args.dynamics))
    plot.plot_2d_dynamics(
        dynamics, 
        save_by=f"{save_by}_dynamics", save=args.save, scale=None,
        **vars(config.dynamics_plot)
    )
    print(f"global lipschitz: {dynamics.global_lipschitz}")

    initial_dist = utils.get_initial_dist(loc=args.initial_dist.loc, variance=args.initial_dist.variance)
    noise_dist = utils.get_noise_dist(loc=args.noise_dist.loc, variance=args.noise_dist.variance)

    q = AmbiguityBall(initial_dist, config.radii.initial)
    noise = AmbiguityBall(noise_dist, config.radii.noise)

    path = multi_step(
        dynamics=dynamics, 
        q=q, 
        noise=noise,
        num_time_steps=num_time_steps,
        use_lagrangian_duality=True,
        num_locs=args.num_locs,
    )

    true_samples = multi_step_empirical(
        dynamics=dynamics,
        p_emp=q.sample(args.num_samples),
        noise=noise,
        num_time_steps=num_time_steps,
    )
    approx_samples = SampledPath({k: path.at(k).sample(args.num_samples) for k in path.ordered_indices})

    plot.plot_2d_ambiguity_balls(
        true_samples, path, 
        save_by=f"{save_by}_path_true", save=args.save, **vars(config.path_plot)
    )
    plot.plot_2d_ambiguity_balls(
        approx_samples, path, 
        save_by=f"{save_by}_path_appr", save=args.save, **vars(config.path_plot)
    )

def analysis(name: str):
    config = paper_configs[name]

    args = parse_arguments(
        dynamics_type=config.dynamics_type,
        dynamics_setting=config.dynamics_setting,
        num_locs=config.num_locs,
        num_samples=config.num_samples,
    )

    hyper_params_analysis(args, name)
    boundary_cond_analysis(args, name)

if __name__ == '__main__':
    torch.manual_seed(0)

    illustrate("double_spiral")

    illustrate("switched_linear_upper_left")

    illustrate("switched_linear_lower_right")

    illustrate("neural_pendulum")

    analysis("quadruple_tank")

    analysis("neural_pendulum")




