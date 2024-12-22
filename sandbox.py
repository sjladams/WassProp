import torch
from typing import Callable, Union

import dynamics
import discretize_distributions as ds
from regions import HyperRectangularVoronoiPartition

import torch

from experiments import multi_step, get_noise_dist, get_initial_dist, single_step_w2_options
from dynamics import get_dynamics
import plot

from utils import load_params, parse_arguments

torch.manual_seed(0)

args = parse_arguments(
    dynamics_type = "MountainCarDynamics",
    num_dims = 2,
    dynamics_setting = 0,
    num_locs = 100,
    num_locs_after_compr=1,
    num_samples = 5000,
    lr = 0.01,
    num_iterations = 1000,
    plot = False,
    optimize_locs=False
)

run_single_step = False
run_multi_step = True

params = load_params(args)

dynamics = get_dynamics(**params)
dist = get_initial_dist(**params)

signature = ds.discretization_generator(dist=dist, num_locs=args.num_locs)

plot.plot_norm_overapproximation(dynamics, signature, 76)