import torch
import matplotlib.patches as patches

import bound_propagation as bp

from duq_via_wasserstein import multi_step, multi_step_empirical, SampledPath, AmbiguitySet
import duq_via_wasserstein.dynamics as dyn

from dynamics import get_dynamics
import plot
from handlers import parse_arguments, param_handler
import utils


class SwitchedLinearDynamics(dyn.Dynamics):
    num_dims = 2
    def __init__(self, **kwargs):
        region = [[-2., -2.], [2., 2.]]

        mat1 = [[0.79, 0.035], [0., 0.825]]
        mat2 = [[0.79, 0.175], [0., 0.825]]
        mat3 = [[0.79, 0.], [0.175, 0.825]]
        mat4 = [[1., 0.2], [-0.2, 1.]]
        mat5 = [[1., -0.2], [0.2, 1.]]
        redun_mat = torch.eye(2)

        mid_region = [[-0.8, -0.8],[0.8, 0.8]]
        mid_block = dyn.PiecewiseAffineBLock(min=mid_region[0], max=mid_region[1], dynamics=dyn.LinearDynamics(weight=redun_mat))

        obs_right = dyn.PiecewiseAffineBLock(min=[1., 1.], max=[2., 2.], dynamics=dyn.LinearDynamics(weight=redun_mat))
        mode2_right = dyn.PiecewiseAffineBLock(min=[mid_region[1][0], 0.25], max=[2., 1.], dynamics=dyn.LinearDynamics(weight=mat2))
        mode5_right = dyn.PiecewiseAffineBLock(min=[mid_region[1][0], -1.], max=[2., 0.25], dynamics=dyn.LinearDynamics(weight=mat5))
        mode1_bottom = dyn.PiecewiseAffineBLock(min=[0., -2.], max=[2., -1.8], dynamics=dyn.LinearDynamics(weight=mat1))
        mode4_bottom = dyn.PiecewiseAffineBLock(min=[0., -1.8], max=[2., -1.], dynamics=dyn.LinearDynamics(weight=mat4))
        mode3 = dyn.PiecewiseAffineBLock(min=[0.3, mid_region[1][1]], max=[1., 2.], dynamics=dyn.LinearDynamics(weight=mat1)) # \todo crux
        mode2_bottom = dyn.PiecewiseAffineBLock(min=[-0.6, -2.], max=[0., mid_region[0][1]], dynamics=dyn.LinearDynamics(weight=mat2))
        mode1_bottom_left = dyn.PiecewiseAffineBLock(min=[-1., -2.], max=[-0.6, mid_region[0][1]], dynamics=dyn.LinearDynamics(weight=mat1))

        mode4_top = dyn.PiecewiseAffineBLock(min=[-1.8, 1.], max=[0.3, 1.8], dynamics=dyn.LinearDynamics(weight=mat4))
        mode2_top = dyn.PiecewiseAffineBLock(min=[-2, 1.8], max=[0.3, 2.], dynamics=dyn.LinearDynamics(weight=mat2))
        mode1_left = dyn.PiecewiseAffineBLock(min=[-2., 0.], max=[-1.8, 1.8], dynamics=dyn.LinearDynamics(weight=mat1))
        mode5_left = dyn.PiecewiseAffineBLock(min=[-1.8, 0.], max=[mid_region[0][0], 1.], dynamics=dyn.LinearDynamics(weight=mat5))
        mode2_left = dyn.PiecewiseAffineBLock(min=[-2., -1.], max=[mid_region[0][0], 0.], dynamics=dyn.LinearDynamics(weight=mat2))
        obs_left = dyn.PiecewiseAffineBLock(min=[-2., -2.], max=[-1., -1.], dynamics=dyn.LinearDynamics(weight=redun_mat))

        redun_mode = dyn.PiecewiseAffineBLock(min=region[0], max=region[1], dynamics=dyn.LinearDynamics(weight=torch.zeros((2,2))))

        super().__init__(
            bp.Clamp(min=torch.as_tensor(region[0]), max=torch.as_tensor(region[1])),
            bp.Parallel(
                obs_right, mode2_right, mode5_right, mode1_bottom,
                mode4_bottom, mode3,
                mode2_bottom, mode1_bottom_left, mode4_top, mode2_top,
                mid_block,
                mode1_left, mode5_left, mode2_left, obs_left,
                redun_mode
            ),
            bp.VectorAdd(), bp.VectorAdd(), bp.VectorAdd(), bp.VectorAdd()
        )

    @property
    def global_lipschitz(self):
        global_lipschitz = []
        for mode in self[1].subnetworks:
            global_lipschitz.append(mode.global_lipschitz)
        return max(global_lipschitz)

get_dynamics.register('SwitchedLinearDynamics', dyn.additive(SwitchedLinearDynamics))

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
    dynamics = get_dynamics(**vars(args))
    plot.plot_2d_dynamics(
        dynamics, 
        patch_creator=patch_creator, text_creator=text_creator, xlim=[-2., 2.], ylim=[-2., 2.], figsize=(12, 12), scale=None, 
        save_by=f"{args.results_folder}dynamics_{args.tag}", save=args.save
    )
    print(f"global lipschitz: {dynamics.global_lipschitz}")
    initial_dist = utils.get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)
    noise_dist = utils.get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    q = AmbiguitySet(initial_dist, 0.01)
    noise = AmbiguitySet(noise_dist, 0.0001)

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
        dict(dynamics_setting= 0, num_time_step=8, tag="upper_left"),
        # dict(dynamics_setting=2, num_time_step=10, tag="lower_right"),
    ]

    store = dict()
    for config in configs:
        dynamics_params = param_handler(
            param_name="dynamics",
            dataset_name=args.dynamics_type,
            setting_tag=config['dynamics_setting']
        )
        args.__dict__.update(vars(dynamics_params))
        args.num_time_steps = config['num_time_step']
        args.tag = config['tag']

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

