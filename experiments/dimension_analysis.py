import os, sys
from collections import defaultdict

import torch
from itertools import product
from typing import Union
import scipy.stats as st

from experiments.plot import plot_dimension_analysis
from wass_prop.dynamics import NNLayerDynamics, StochasticDynamics

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "src"))

import discretize_distributions as dd
from discretize_distributions.distributions import MultivariateNormal, MixtureMultivariateNormal
from wass_prop import AmbiguityBall, multi_step
from dynamics import AdditiveNoiseDynamics

import time
import tracemalloc

def aggregate_stats(data):
    # Group tensors by (dimension, num_locs)
    grouped = defaultdict(lambda: {"w2": [], "exec_time": [], "memory": []})

    for (dim, locs, seed), vals in data.items():
        for key in ["w2", "exec_time", "memory"]:
            # Convert scalars to tensors if they aren't already
            grouped[(dim, locs)][key].append(torch.as_tensor(vals[key], dtype=torch.float32))

    means = {}
    stds = {}

    for key, fields in grouped.items():
        means[key] = {}
        stds[key] = {}

        for field, values in fields.items():
            stack = torch.stack(values)
            means[key][field] = stack.mean().item()
            stds[key][field] = stack.std(unbiased=True).item()

    return means, stds

def analyze_discretization(distribution: Union[MultivariateNormal, MixtureMultivariateNormal],
                           num_locs: int):
    tracemalloc.start()
    t0 = time.perf_counter()

    scheme = dd.generate_scheme(distribution, scheme_size=num_locs)
    disc, w2 = dd.discretize(distribution, scheme)

    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    execution_time = elapsed
    memory = peak / 1024 ** 2 # In MB

    return { "w2": w2.item(), "exec_time": execution_time, "memory": memory }

def analyze_propagation(dynamics: StochasticDynamics,
                        p: AmbiguityBall,
                        noise: AmbiguityBall,
                        num_locs: int,
                        num_steps: int):

    tracemalloc.start()
    t0 = time.perf_counter()

    path = multi_step(dynamics=dynamics, q=p, noise=noise, num_time_steps=num_steps, num_locs=num_locs, print_progress=False)

    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    execution_time = elapsed / num_steps
    memory = peak / 1024 ** 2  # In MB

    return {"w2": path.at(0).w2.item(), "exec_time": execution_time, "memory": memory}

def sample_covariance(dimension: int, low=1e-3, high=1e-1):
    diag_vals = torch.rand(dimension) * (high - low) + low
    return torch.diag(diag_vals)

def sample_weight(dimension: int):
    A = torch.randn(dimension, dimension)
    A = 0.5 * (A + A.T)

    eig_max = torch.linalg.eigvalsh(A).max()
    if eig_max >= 1:
        A = A / (eig_max + 1e-8)
    return A

def get_normalized_variance(dimension: int, p: float = 0.95):
    chi2_quantile = st.chi2.ppf(p, df=dimension)
    return 1.0 / chi2_quantile

def manifold_distributions(dimension, num_dists, small=1e-6, large=1e-2):
    dists = []
    small_counts = torch.linspace(0, dimension, num_dists, dtype=torch.int32)

    for k, n_small in enumerate(small_counts):
        diag = torch.full((dimension,), large)

        if n_small > 0:
            diag[:n_small.item()] = small

        cov = torch.diag(diag)
        mvn = MultivariateNormal(loc=torch.zeros(dimension), covariance_matrix=cov)
        dists.append(mvn)

    return dists

if __name__ == '__main__':
    torch.manual_seed(0)

    ###################################################
    # Experiment: quantization and propagation        #
    ###################################################
    # Set parameters
    dimensions = [2, 3, 10, 25, 50, 75, 100]
    nums_locs = [10, 100, 1000]
    num_random_seeds = 10

    num_steps = 1

    # Collect data
    data_quant, data_prop = {}, {}
    for dimension, num_locs in product(dimensions, nums_locs):
        for random_seed in range(num_random_seeds):

            # Quantization
            distribution = MultivariateNormal(loc=torch.zeros(dimension), covariance_matrix=get_normalized_variance(dimension) * torch.eye(dimension))
            data_quant[(dimension, num_locs, random_seed)] = analyze_discretization(distribution=distribution, num_locs=num_locs)

            # Propagation
            dynamics = AdditiveNoiseDynamics(state_dynamics=NNLayerDynamics(weight=sample_weight(dimension), bias=None)) # random NN layer

            p = AmbiguityBall(
                center=distribution,
                radius=0.1
            )
            noise = AmbiguityBall(
                center=MultivariateNormal(loc=torch.zeros(dimension), covariance_matrix=torch.eye(dimension) * 1e-4),
                radius=0.01
            )

            data_prop[(dimension, num_locs, random_seed)] = analyze_propagation(dynamics=dynamics, p=p, noise=noise, num_locs=num_locs, num_steps=num_steps)

    means_quant, stds_quant = aggregate_stats(data_quant)
    means_prop, stds_prop = aggregate_stats(data_prop)

    plot_dimension_analysis(means_quant, stds_quant)
    plot_dimension_analysis(means_prop, stds_prop)

    #######################################################
    # Experiment: probability concentration in a manifold #
    #######################################################

    dimension = 100
    nums_locs = [10000]

    num_dists = 5
    small = 1e-5
    large = 1e-2

    distributions = manifold_distributions(dimension, num_dists=num_dists, small=small, large=large)

    # Collect data
    data_quant = {}
    for distribution in distributions:

        manifold_dim = (distribution.covariance_matrix.diag() == small).sum().item()
        for num_locs in nums_locs:
            # Quantization
            data_quant[(manifold_dim, num_locs)] = analyze_discretization(distribution=distribution, num_locs=num_locs)

    plot_dimension_analysis(data_quant, x_axis_title="Dimension of manifold")