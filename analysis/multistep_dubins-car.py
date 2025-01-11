import torch

from experiments import multi_step, get_noise_dist, get_initial_dist, single_step_w2_options
from dynamics import get_dynamics
import plot

from utils import load_params, parse_arguments

def multistep_approximation(dynamics_type, num_dims, dyn_setting, num_locs):
    args = parse_arguments(
        dynamics_type=dynamics_type,
        num_dims=num_dims,
        dynamics_setting=dyn_setting,
        num_locs=num_locs,
        num_locs_after_compr=num_locs,
        num_samples=5000,
        lr=0.01,
        num_iterations=1000,
        plot=False,
        optimize_locs=False
    )

    run_multi_step = True

    params = load_params(args)

    dynamics = get_dynamics(**params)
    initial_dist = get_initial_dist(**params)
    noise_dist = get_noise_dist(**params)

    if run_multi_step:
        w2_bounds, samples = multi_step(
            dynamics=dynamics,
            noise_dist=noise_dist,
            q=initial_dist,
            num_time_steps=10,
            run_independent_coupling=False,
            run_lagrangian_duality=True,
            run_empirical=True,
            **params
        )
    plot.plot_multi_step(dynamics, samples)

if __name__ == '__main__':
    torch.manual_seed(0)

    dynamics_type = 'DiscreteDubinsCarDynamics'
    num_dims = 3
    num_locs = 100
    dyn_setting = 0

    multistep_approximation(dynamics_type, num_dims, dyn_setting, num_locs)



