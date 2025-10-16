import torch
import discretize_distributions as dd

import duq_via_wasserstein.wasserstein as wasserstein

from dynamics import get_dynamics
import plot
from handlers import parse_arguments
import utils


def quantization_example(dynamics_type, setting, nums_locs):
    args = parse_arguments(
            dynamics_type=dynamics_type,
            dynamics_setting=setting,
    )

    signatures, bounds = [], []
    for num_locs in nums_locs:
        args.num_locs = num_locs
        
        dynamics = get_dynamics(**vars(args))
        initial_dist =  utils.get_initial_dist(args.loc_initial_dist, args.variance_initial_dist)

        scheme = dd.generate_scheme(dist=initial_dist, scheme_size=num_locs)
        signature, w2 = dd.discretize(initial_dist, scheme)
        signatures.append(signature)

        fn_sq_w2_f_q__f_disc_q = wasserstein.get_fn_sq_w2_f_q__f_disc_q(initial_dist, signature, dynamics.state_dynamics)
        bound = fn_sq_w2_f_q__f_disc_q().sqrt()
        bounds.append(bound)

    plot.plot_signatures(dynamics.state_dynamics, initial_dist, signatures, bounds)

if __name__ == '__main__':
    torch.manual_seed(0)

    dynamics_type = 'SigmoidDynamics'
    nums_locs = [5, 10, 100]
    setting = 0

    quantization_example(dynamics_type, setting, nums_locs)