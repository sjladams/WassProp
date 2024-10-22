import torch
import argparse

from setup_experiments import single_step, multi_step
from dynamics import get_dynamics
import plotting

def parse_arguments():
    parser = argparse.ArgumentParser(description='Setup experiments for dynamics.')
    parser.add_argument('--dynamics_type',
                        type=str,
                        choices=['GaussianDynamics1d', 'ChaoticDynamics', 'LinearDynamics'],
                        default='ChaoticDynamics',
                        help='Type of dynamics to use.')
    parser.add_argument('--dynamics_setting',
                        type=int,
                        default=0,
                        help='Parameters for the dynamics as a dictionary string.')
    parser.add_argument('--loc_noise_dist',
                        type=float,
                        nargs='+',
                        default=[0.],
                        help='Mean of the noise distribution.')
    parser.add_argument('--variance_noise_dist',
                        type=float,
                        nargs='+',
                        default=[0.3],
                        help='Variance of the noise distribution.')
    parser.add_argument('--loc_initial_dist',
                        type=float,
                        nargs='+',
                        default=[4.0],
                        help='Mean of the initial distribution.')
    parser.add_argument('--variance_initial_dist',
                        type=float,
                        nargs='+',
                        default=[2.],
                        help='Variance of the initial distribution.')
    parser.add_argument('--nr_signature_points',
                        type=int,
                        default=10,
                        help='Number of signature points.')
    parser.add_argument('--num_samples',
                        type=int,
                        default=1000,
                        help='Number of samples.')
    parser.add_argument('--plot',
                        type=bool,
                        default=True,
                        help='Plot the dynamics and distributions.')
    return parser.parse_args()

# \Todo include as json file:
PARAMS_DYNAMICS = {
    'GaussianDynamics1d':
        {'loc': torch.zeros(1), 'scale': torch.ones(1)},
    'ChaoticDynamics':
        {'r': 4},
    'LinearDynamics':
        {
            0: {'mat': torch.tensor([[0.5]])},
            1: {'mat': torch.tensor([[0.5, 0.], [0., 0.5]])}
        }
}

if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments()
    dynamics = get_dynamics(
        args.dynamics_type,
        **(PARAMS_DYNAMICS[args.dynamics_type][args.dynamics_setting] if args.dynamics_setting in PARAMS_DYNAMICS[args.dynamics_type] else
           PARAMS_DYNAMICS[args.dynamics_type]))

    # Run single step experiment:
    w2_p__q_options = [0., 0.1, 0.5, 1.]
    w2_bounds, tag = single_step(
        dynamics=dynamics,
        loc_noise_dist=torch.tensor(args.loc_noise_dist),
        covariance_noise_dist=torch.diag(torch.tensor(args.variance_noise_dist)),
        loc_initial_dist=torch.tensor(args.loc_initial_dist),
        covariance_initial_dist=torch.diag(torch.tensor(args.variance_initial_dist)),
        num_samples=args.num_samples,
        num_signature_points=args.nr_signature_points,
        run_type1=False,
        w2_p__q_options=w2_p__q_options,
        plot=args.plot
    )

    plotting.plot_single_step(dynamics, w2_bounds, tag, w2_p__q_options)

    # Run multi step experiment:
    w2_bounds, tag = multi_step(
        dynamics= dynamics,
        loc_noise_dist= torch.tensor(args.loc_noise_dist),
        covariance_noise_dist= torch.diag(torch.tensor(args.variance_noise_dist)),
        loc_initial_dist= torch.tensor(args.loc_initial_dist),
        covariance_initial_dist= torch.diag(torch.tensor(args.variance_initial_dist)),
        num_samples=args.num_samples,
        num_time_steps=10,
        num_signature_points=args.nr_signature_points,
        run_type1=False,
        plot=args.plot
    )

    plotting.plot_multi_step(dynamics, w2_bounds, tag)





