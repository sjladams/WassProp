import torch
from experiments import multi_step, get_noise_dist, get_initial_dist, single_step_w2_options
from dynamics import get_dynamics
import plot
from utils import load_params, parse_arguments
from utils_distributions import quantize


def initial_step(dynamics_type, num_dims, dyn_setting, nums_locs):

    signatures, bounds = [], []

    for num_locs in nums_locs:
        args = parse_arguments(
            dynamics_type=dynamics_type,
            num_dims=num_dims,
            dynamics_setting=dyn_setting,
            num_locs=num_locs,
            num_locs_after_compr=num_locs,
            num_samples=5000,
            optimize_locs=False
        )

        params = load_params(args)

        dynamics = get_dynamics(**params)
        initial_dist = get_initial_dist(**params)
        noise_dist = get_noise_dist(**params)

        signature, w2 = quantize(initial_dist, num_locs)
        signatures.append(signature)

        w2_bounds = single_step_w2_options(
            dynamics=dynamics,
            noise_dist=noise_dist,
            q=initial_dist,
            w2_p__q_options=[0.],
            run_independent_coupling=False,
            run_lagrangian_duality=True,
            **params
        )

        bound = w2_bounds[0.0]['lagrangian_duality'].item()
        bounds.append(bound)

    plot.plot_signatures(dynamics, initial_dist, signatures, bounds)

if __name__ == '__main__':
    torch.manual_seed(0)

    dynamics_type = 'SigmoidDynamics'
    num_dims = 1
    nums_locs = [5, 10, 100]
    dyn_setting = 0

    initial_step(dynamics_type, num_dims, dyn_setting, nums_locs)