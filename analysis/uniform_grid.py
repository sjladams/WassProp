import math

import torch
import wasserstein
from bound import global_lbp_sq_norm_fx_fc, global_ibp_sq_norm_fx_fc
from experiments import multi_step, get_noise_dist, get_initial_dist, single_step_w2_options
from dynamics import get_dynamics
import plot
from regions import HyperRectangularVoronoiPartition
from utils import load_params, parse_arguments
from utils_distributions import quantize
from torch.distributions import Normal
import discretize_distributions as ds


def make_uniform_grid(points):
    n, d = points.shape

    # Compute the approximate number of grid points per dimension
    grid_size = math.ceil(n ** (1 / d))  # Number of points per dimension
    total_grid_points = grid_size ** d  # Total points in the full grid

    # Generate equally spaced values for each dimension
    min_vals, _ = torch.min(points, dim=0)
    max_vals, _ = torch.max(points, dim=0)

    linspaces = [torch.linspace(min_vals[i], max_vals[i], steps=grid_size) for i in range(d)]

    # Create a full grid using meshgrid
    grids = torch.meshgrid(*linspaces, indexing="ij")  # Ensure correct indexing
    grid_points = torch.stack([g.flatten() for g in grids], dim=1)  # Flatten into (total_grid_points, d)

    # Remove duplicate points using torch.unique
    unique_grid_points = torch.unique(grid_points, dim=0)

    # Select the first n points to maintain the original count (if needed)
    #if unique_grid_points.shape[0] > n:
    #    unique_grid_points = unique_grid_points[:n]

    return unique_grid_points

def gaussian_hypercube_prob(dist, lower, upper):

    standard_normal = Normal(0, 1)
    mu = dist.loc
    sigma = dist.stddev

    lower_std = (lower - mu) / sigma
    upper_std = (upper - mu) / sigma

    cdf_diff = standard_normal.cdf(upper_std) - standard_normal.cdf(lower_std)
    probabilities = cdf_diff.prod(dim=1)

    return probabilities

def compare_grids(dynamics_type, num_dims, dyn_setting, nums_locs):

    signatures_optimal, bounds_optimal = [], []
    signatures_uniform, bounds_uniform = [], []

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
        initial_dist = get_initial_dist(**params)

        signature, w2 = quantize(initial_dist, num_locs)
        signatures_optimal.append(signature)

        fn_sq_w2_f_q__f_disc_q = wasserstein.get_fn_sq_w2_f_q__f_disc_q(signature, dynamics)
        bound = fn_sq_w2_f_q__f_disc_q().sqrt()
        bounds_optimal.append(bound)

        # Compare with uniform grid
        locs = make_uniform_grid(signature.locs)

        voronoi_partition = HyperRectangularVoronoiPartition(locs)
        alpha = global_lbp_sq_norm_fx_fc(dynamics, locs)
        beta = global_ibp_sq_norm_fx_fc(dynamics, locs).upper.squeeze(-1)

        sq_norm_2nd_moment = wasserstein.compute_sq_norm_2nd_moment(initial_dist, voronoi_partition, locs)
        probs = gaussian_hypercube_prob(initial_dist, voronoi_partition.lower, voronoi_partition.upper)

        w2_alpha_or_beta = torch.min(sq_norm_2nd_moment * alpha, beta)
        bound =  torch.einsum('...i,...i->...', w2_alpha_or_beta, probs).sqrt()
        bounds_uniform.append(bound)

        signatures_uniform.append(ds.CategoricalFloat(probs, locs))

    print(bounds_optimal)
    print(bounds_uniform)

    plot.plot_signatures(dynamics, initial_dist, signatures_optimal, bounds_optimal)
    plot.plot_signatures(dynamics, initial_dist, signatures_uniform, bounds_uniform)

if __name__ == '__main__':
    torch.manual_seed(0)

    dynamics_type = 'LinearDiagonalBoundedDynamics'
    num_dims = 1
    nums_locs = [5, 10, 100]
    dyn_setting = 0

    compare_grids(dynamics_type, num_dims, dyn_setting, nums_locs)