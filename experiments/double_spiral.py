import torch
import bound_propagation as bp

from duq_via_wasserstein import multi_step, multi_step_empirical, SampledPath, AmbiguitySet
import duq_via_wasserstein.dynamics as dyn

from dynamics import get_dynamics
import plot
from handlers import parse_arguments
import utils


class DoubleSpiral2dDynamics(dyn.Dynamics):
    num_dims = 2
    def __init__(self, **kwargs):
        region = [[-2., -0.75], [2., 1.25]]

        weight_left = utils.rot_mat(theta=torch.pi / 8., rho=0.8, delta=0.)
        weight_right = utils.rot_mat(theta=-torch.pi / 8., rho=0.8, delta=0.)
        bias_left = (torch.eye(2) - weight_left) @ torch.tensor([-1.25, -1.0])
        bias_right = (torch.eye(2) - weight_right) @ torch.tensor([1.25, -1.0])

        mode_left = dyn.PiecewiseAffineBLock(min=torch.tensor(region[0]), max=torch.tensor([0., region[1][1]]),
                                         dynamics=dyn.LinearDynamics(weight=weight_left, bias=bias_left))
        mode_right = dyn.PiecewiseAffineBLock(min=torch.tensor([0., region[0][1]]), max=torch.tensor(region[1]),
                                          dynamics=dyn.LinearDynamics(weight=weight_right, bias=bias_right))

        super().__init__(
            bp.Clamp(min=torch.tensor(region[0]), max=torch.tensor(region[1])),
            bp.Parallel(mode_left, mode_right),
            bp.VectorAdd()
        )

    @property
    def global_lipschitz(self):
        global_lipschitz = []
        for mode in self[1].subnetworks:
            global_lipschitz.append(mode.global_lipschitz)
        return max(global_lipschitz)

get_dynamics.register('DoubleSpiral2dDynamics', dyn.additive(DoubleSpiral2dDynamics))


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        dynamics_type = "DoubleSpiral2dDynamics",
        dynamics_setting = 0,
        num_locs = 10,
        num_samples = 100,
        save = False
    )

    num_time_steps = 10

    save_by = f"{args.results_folder}double_spiral"
    dynamics = get_dynamics(**vars(args))
    plot.plot_2d_dynamics(
        dynamics, 
        xlim= [-2.0, 2.0], ylim=[-1.0, 1.0], figsize=(13, 8), scale=None, 
        save_by=f"{save_by}_dynamics", save=args.save
    )
    print(f"global lipschitz: {dynamics.global_lipschitz}")

    initial_dist = utils.get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)
    noise_dist = utils.get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    q = AmbiguitySet(initial_dist, 0.1)
    noise = AmbiguitySet(noise_dist, 0.01)

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
        num_samples=args.num_samples,
    )
    approx_samples = SampledPath({k: path.at(k).sample(args.num_samples) for k in path.ordered_indices})

    plot.plot_2d_ambiguity_balls(
        true_samples, path, 
        xlim=[-1.0, 1.0], ylim=[-1., 1.0], figsize=(9, 8), 
        save_by=f"{save_by}_path_true", save=args.save
    )
    plot.plot_2d_ambiguity_balls(
        approx_samples, path, 
        xlim=[-1.0, 1.0], ylim=[-1., 1.0], figsize=(9, 8), 
        save_by=f"{save_by}_path_appr", save=args.save
    )
    





