import torch
import wasserstein
from utils_distributions import get_initial_dist
from dynamics import get_dynamics
import plot
from utils import load_params, parse_arguments
from utils_distributions import quantize


def quantization_example(dynamics_type, num_dims, dyn_setting, nums_locs):

    signatures, bounds = [], []

    for num_locs in nums_locs:

        args = parse_arguments(
            dynamics_type=dynamics_type,
            num_dims=num_dims,
            dynamics_setting=dyn_setting,
            num_locs=num_locs,
            num_locs_after_compr=num_locs,
        )
        params = load_params(args)

        dynamics = get_dynamics(**params)
        distribution = get_initial_dist(**params)

        signature, w2 = quantize(distribution, num_locs)
        signatures.append(signature)

        fn_sq_w2_f_q__f_disc_q = wasserstein.get_fn_sq_w2_f_q__f_disc_q(signature, dynamics.state_dynamics)
        bound = fn_sq_w2_f_q__f_disc_q().sqrt()
        bounds.append(bound)

    plot.plot_signatures(dynamics.state_dynamics, distribution, signatures, bounds)

if __name__ == '__main__':
    torch.manual_seed(0)

    dynamics_type = 'SigmoidDynamics'
    num_dims = 1
    nums_locs = [5, 10, 100]
    dyn_setting = 0

    quantization_example(dynamics_type, num_dims, dyn_setting, nums_locs)