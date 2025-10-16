import torch
import os

from duq_via_wasserstein import multi_step, multi_step_empirical, SampledPath, AmbiguitySet
import duq_via_wasserstein.dynamics as dyn

from dynamics import get_dynamics, SigmoidDynamics, TanhDynamics
import plot as plot
from handlers import parse_arguments
import utils


class NeuralPendulumDynamics(dyn.Dynamics, dyn.CompositionalStructure):
    def __init__(self, data_folder: str, activation: str, **kwargs):
        self.num_dims = 2

        state_dict = torch.load(f'{data_folder}{activation}_model_weights_pendulum.pth', weights_only=True)

        weight_fc1 = state_dict["fc1.weight"]
        bias_fc1 = state_dict["fc1.bias"]
        weight_fc2 = state_dict["fc2.weight"]
        bias_fc2 = state_dict["fc2.bias"]
        weight_fc3 = state_dict["fc3.weight"]
        bias_fc3 = state_dict["fc3.bias"]

        if activation == 'sigmoid':
            ActivationDynamics = SigmoidDynamics
        elif activation == 'tanh':
            ActivationDynamics = TanhDynamics
        else:
            raise NotImplementedError(f"Activation {activation} not implemented.")

        super().__init__(
            dyn.LinearDynamics(weight_fc1, bias_fc1),
            ActivationDynamics(bias_fc1.size(0)),
            dyn.LinearDynamics(weight_fc2, bias_fc2),
            ActivationDynamics(bias_fc2.size(0)),
            dyn.LinearDynamics(weight_fc3, bias_fc3)
        )

    @property
    def global_lipschitz(self):
        return torch.tensor([module.global_lipschitz for module in self]).prod()

get_dynamics.register('NeuralPendulumDynamics', dyn.additive(NeuralPendulumDynamics))


def illustrate():
    args = parse_arguments(
        dynamics_type = "NeuralPendulumDynamics",
        dynamics_setting = 0,
        num_locs = 100,
        num_samples = 100,
        save = False
    )

    num_time_steps = 20

    save_by = f"{args.results_folder}neural_pendulum"
    dynamics = get_dynamics(**vars(args))
    plot.plot_2d_dynamics(
        dynamics, 
        xlim=[-2.0, 1.8], ylim=[-1.8, 2.0], figsize=(12,12), 
        save_by=f"{save_by}_dynamics", save=args.save
    )
    print(f"global lipschitz: {dynamics.global_lipschitz}")

    initial_dist = utils.get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)
    noise_dist = utils.get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    q = AmbiguitySet(initial_dist, 0.001)
    noise = AmbiguitySet(noise_dist, 0.001)

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
        xlim=[-2.1, 0.8], ylim=[-1.0, 1.5], figsize=(12,12), 
        save_by=f"{save_by}_path_true", save=args.save
    )

    plot.plot_2d_ambiguity_balls(
        approx_samples, path,
        xlim=[-2.1, 0.8], ylim=[-1.0, 1.5], figsize=(12,12), 
        save_by=f"{save_by}_path_approx", save=args.save
    )


def quantitative_analysis():
    args = parse_arguments(
            dynamics_type="NeuralPendulumDynamics",
            dynamics_setting=0,
            num_locs=100,
            num_samples=100,
            save=False
        )
    
    name = "neural_pendulum"

    utils.hyper_params_analysis(args, name, w2_p__q=0.01, w2_noise_dist=0.01)
    utils.boundary_cond_analysis(args, name)


if __name__ == '__main__':
    torch.manual_seed(0)

    illustrate()

    # quantitative_analysis()

