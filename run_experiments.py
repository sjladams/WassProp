import torch


from setup_experiments import single_step, multi_step
from dynamics import get_dynamics
import plotting

from utils import load_params, parse_arguments


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        dynamics_type = 'LinearDynamics',
        num_dims = 2,
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

    plotting.plot_single_step(dynamics, w2_bounds, tag, w2_p__q_options)

    # # Run multi step experiment:
    # w2_bounds, tag = multi_step(
    #     dynamics= dynamics,
    #     loc_noise_dist= torch.tensor(args.loc_noise_dist),
    #     covariance_noise_dist= torch.diag(torch.tensor(args.variance_noise_dist)),
    #     loc_initial_dist= torch.tensor(args.loc_initial_dist),
    #     covariance_initial_dist= torch.diag(torch.tensor(args.variance_initial_dist)),
    #     num_samples=args.num_samples,
    #     num_time_steps=10,
    #     num_signature_points=args.nr_signature_points,
    #     run_type1=False,
    #     plot=args.plot
    # )
    #
    # plotting.plot_multi_step(dynamics, w2_bounds, tag)





