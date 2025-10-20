import torch
import bound_propagation as bp

from duq_via_wasserstein import multi_step, multi_step_empirical, SampledPath, AmbiguityBall
from duq_via_wasserstein.dynamics import Dynamics, IndicatorDynamics, LinearDynamics, AdditiveNoiseDynamics

from handlers import parse_arguments
import plot
import utils


class DoubleSpiral2dDynamics(Dynamics):
    def __init__(self):
        region_left = torch.tensor([[-2., -0.75], [0., 1.25]])
        weight_left = utils.rot_mat(theta=torch.pi / 8., rho=0.8, delta=0.)
        bias_left = (torch.eye(2) - weight_left) @ torch.tensor([-1.25, -1.0])

        region_right = torch.tensor([[0., -0.75], [2., 1.25]])
        weight_right = utils.rot_mat(theta=-torch.pi / 8., rho=0.8, delta=0.)
        bias_right = (torch.eye(2) - weight_right) @ torch.tensor([1.25, -1.0])

        mode_left = IndicatorDynamics(
            lower=region_left[0], 
            upper=region_left[1],
            dynamics=LinearDynamics(weight=weight_left, bias=bias_left)
        )
        mode_right = IndicatorDynamics(
            lower=region_right[0], 
            upper=region_right[1],
            dynamics=LinearDynamics(weight=weight_right, bias=bias_right)
        )

        super().__init__(
            num_dims=2,
            modules=[
                bp.Clamp(min=region_left[0], max=region_right[1]),
                bp.Parallel(mode_left, mode_right),
                bp.VectorAdd()
            ]
        )

    @property
    def global_lipschitz(self):
        global_lipschitz = []
        for mode in self[1].subnetworks:
            global_lipschitz.append(mode.global_lipschitz)
        return max(global_lipschitz)


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
    dynamics = AdditiveNoiseDynamics(DoubleSpiral2dDynamics())
    plot.plot_2d_dynamics(
        dynamics, 
        xlim= [-2.0, 2.0], ylim=[-1.0, 1.0], figsize=(13, 8), scale=None, 
        save_by=f"{save_by}_dynamics", save=args.save
    )
    print(f"global lipschitz: {dynamics.global_lipschitz}")

    initial_dist = utils.get_initial_dist(loc=args.initial_dist.loc, variance=args.initial_dist.variance)
    noise_dist = utils.get_noise_dist(loc=args.noise_dist.loc, variance=args.noise_dist.variance)

    q = AmbiguityBall(initial_dist, 0.1)
    noise = AmbiguityBall(noise_dist, 0.01)

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
        xlim=[-1.0, 1.0], ylim=[-1., 1.0], figsize=(9, 8), 
        save_by=f"{save_by}_path_true", save=args.save
    )
    plot.plot_2d_ambiguity_balls(
        approx_samples, path, 
        xlim=[-1.0, 1.0], ylim=[-1., 1.0], figsize=(9, 8), 
        save_by=f"{save_by}_path_appr", save=args.save
    )
    





