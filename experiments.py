import torch

from main import single_step, multi_step
from dynamics import get_dynamics
import plot

from utils import load_params, parse_arguments


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        dynamics_type = 'BoundedLinearDiagonalDynamics',
        num_dims = 1,
        dynamics_setting = 0,
        num_locs = 100,
        num_samples = 1000,
        lr = 0.01,
        num_iterations = 1000,
        plot = False,
        prob_shell = 0.0001,
    )

    params = load_params(args)

    dynamics = get_dynamics(**params)

    # Run single step experiment:
    w2_p__q_options = [0., 0.1, 0.5, 1.0]
    w2_bounds, tag = single_step(
        dynamics=dynamics,
        run_triangle_type1=False,
        w2_p__q_options=w2_p__q_options,
        **params
    )

    plot.plot_single_step(dynamics, w2_bounds, tag, w2_p__q_options)

