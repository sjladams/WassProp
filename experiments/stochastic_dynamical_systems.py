import torch
from experiments import multi_step
from utils_distributions import get_noise_dist, get_initial_dist
from dynamics import get_dynamics
import experiments.plot as plot
from experiments.utils import parse_arguments

def multistep_approximation(dynamics_type, dyn_setting, num_locs):
    args = parse_arguments(
        dynamics_type=dynamics_type,
        dynamics_setting=dyn_setting,
        num_locs=num_locs,
        # num_locs_after_compr=num_locs,
        num_samples=5000
    )

    dynamics = get_dynamics(**vars(args))
    initial_dist = get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)
    noise_dist = get_noise_dist(args.loc_noise_dist, args.variance_noise_dist)

    _, _, samples = multi_step(
        dynamics=dynamics,
        noise_dist=noise_dist,
        q=initial_dist,
        num_time_steps=10,
        run_lagrangian_duality=True,
        run_empirical=True,
        propagate_via_gmm=False,
        num_samples=args.num_samples,
        num_locs=args.num_locs
    )
    plot.plot_multi_step(dynamics, samples, type=dynamics_type)

if __name__ == '__main__':
    torch.manual_seed(0)

    dynamics_type = 'MountainCarDynamics'
    num_locs = 100
    dyn_setting = 0

    multistep_approximation(dynamics_type, dyn_setting, num_locs)