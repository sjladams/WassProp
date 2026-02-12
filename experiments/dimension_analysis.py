import os, sys
from itertools import product
from typing import Union

from experiments.plot import plot_dimension_analysis
from wass_prop.dynamics import NNLayerDynamics, StochasticDynamics

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "src"))

import torch
import matplotlib.pyplot as plt

from discretize_distributions.distributions import MultivariateNormal, MixtureMultivariateNormal
from wass_prop import AmbiguityBall, multi_step
from dynamics import AdditiveNoiseDynamics

import discretize_distributions as dd

import time
import tracemalloc

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
                        num_locs: int):

    tracemalloc.start()
    t0 = time.perf_counter()

    path = multi_step(dynamics=dynamics, q=p, noise=noise, num_time_steps=1, num_locs=num_locs, print_progress=False)

    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    execution_time = elapsed
    memory = peak / 1024 ** 2  # In MB

    return {"w2": path.at(0).w2.item(), "exec_time": execution_time, "memory": memory}

if __name__ == '__main__':
    torch.manual_seed(0)

    ### Parameters
    distributions = [
        MultivariateNormal(loc=torch.zeros(2), covariance_matrix=torch.eye(2) * 1e-3),
        MultivariateNormal(loc=torch.zeros(5), covariance_matrix=torch.eye(5) * 1e-3),
        MultivariateNormal(loc=torch.zeros(10), covariance_matrix=torch.eye(10) * 1e-3),
        MultivariateNormal(loc=torch.zeros(50), covariance_matrix=torch.eye(50) * 1e-3),
        MultivariateNormal(loc=torch.zeros(100), covariance_matrix=torch.eye(100) * 1e-3),
    ]

    nums_locs = [10, 100, 1000, 10000]

    weights = [
        torch.randn((2, 2)),
        torch.randn((5, 5)),
        torch.randn((10, 10)),
        torch.randn((50, 50)),
        torch.randn((100, 100)),
    ]

    # Quantization
    results_quantization = {}
    for distribution, num_locs in product(distributions, nums_locs):
        results_quantization[(distribution.loc.shape[0], num_locs)] = analyze_discretization(distribution=distribution, num_locs=num_locs)

    # Propagation
    results_propagation = {}
    for (distribution, weight), num_locs in product(zip(distributions, weights), nums_locs):
        dynamics = AdditiveNoiseDynamics(state_dynamics=NNLayerDynamics(weight=weight, bias=None))

        p = AmbiguityBall(
            center=distribution,
            radius=0.1
        )
        noise = AmbiguityBall(
            center=MultivariateNormal(loc=torch.zeros(distribution.loc.shape[0]), covariance_matrix=torch.eye(distribution.loc.shape[0]) * 1e-4),
            radius=0.01
        )

        results_propagation[(distribution.loc.shape[0], num_locs)] = analyze_propagation(dynamics=dynamics, p=p, noise=noise, num_locs=num_locs)

    plot_dimension_analysis(results_quantization, results_propagation)

