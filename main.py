import torch

from experiments import multi_step, get_noise_dist, get_initial_dist, single_step_w2_options
from dynamics import get_dynamics
import plot

from utils import load_params, parse_arguments


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        dynamics_type = "SineDynamics",
        num_dims = 1,
        dynamics_setting = 0,
        num_locs = 10,
        num_locs_after_compr=10,
        num_samples = 1000,
        lr = 0.01,
        num_iterations = 1000,
        plot = False,
        additive_gaussian_noise = False,
        optimize_locs=False
    )

    run_single_step = False
    run_multi_step = True

    params = load_params(args)

    dynamics = get_dynamics(**params)
    initial_dist = get_initial_dist(**params)
    noise_dist = get_noise_dist(**params)

    if run_single_step:
        w2_bounds = single_step_w2_options(
            dynamics=dynamics,
            noise_dist=noise_dist,
            q=initial_dist,
            w2_p__q_options=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0],
            run_independent_coupling=False,
            run_lagrangian_duality=True,
            **params
        )
        plot.plot_single_step(dynamics, w2_bounds, **params)
    elif run_multi_step:
        w2_bounds, samples = multi_step(
            dynamics=dynamics,
            noise_dist=noise_dist,
            q=initial_dist,
            num_time_steps=10,
            run_independent_coupling=False,
            run_lagrangian_duality=True,
            run_empirical=False,
            **params
        )
        plot.plot_multi_step(dynamics, samples)
