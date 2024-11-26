import torch

from experiments import multi_step, get_noise_dist, get_initial_dist, single_step_w2_options
from dynamics import get_dynamics
import plot

from utils import load_params, parse_arguments


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        dynamics_type = 'BoundedLinearDiagonalDynamics',
        num_dims = 1,
        dynamics_setting = 0,
        num_locs = 10,
        num_samples = 1000,
        lr = 0.01,
        num_iterations = 1000,
        plot = False,
    )

    params = load_params(args)

    dynamics = get_dynamics(**params)
    initial_dist = get_initial_dist(**params)
    noise_dist = get_noise_dist(**params)

    # Run single step experiment:
    w2_p__q_options = [0., 0.1, 0.5, 1.0]
    w2_bounds = single_step_w2_options(
        dynamics=dynamics,
        noise_dist=noise_dist,
        q=initial_dist,
        w2_p__q_options=w2_p__q_options,
        run_independent_coupling=True,
        run_lagrangian_duality=True,
        **params
    )

    plot.plot_single_step(dynamics, w2_bounds, w2_p__q_options, **params)

    # Run multi step experiment
    w2_bounds, samples = multi_step(
        dynamics=dynamics,
        noise_dist=noise_dist,
        q=initial_dist,
        num_time_steps=2,
        **params
    )