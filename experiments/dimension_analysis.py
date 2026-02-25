import os, sys
from collections import defaultdict

import torch
from itertools import product
from typing import Union, Optional, List
import scipy.stats as st

from wass_prop.dynamics import NNLayerDynamics, StochasticDynamics

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "src"))

import discretize_distributions as dd
from discretize_distributions.distributions import MultivariateNormal, MixtureMultivariateNormal
from wass_prop import AmbiguityBall, multi_step
from dynamics import AdditiveNoiseDynamics

import time
import tracemalloc

import matplotlib.pyplot as plt

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


def plot_dimension_analysis(
    data_means: dict,
    data_stds: Optional[dict] = None,
    x_axis_title: str = r"Dimension $d$",
    metrics_to_plot: List[str] = ("w2", "exec_time", "memory"),
    data_means_comp: Optional[dict] = None,
    data_stds_comp: Optional[dict] = None,
    x_axis_title_comp: Optional[str] = None
):
    metric_labels = {
        "w2": r"$2$-Wasserstein bound",
        "exec_time": "Time (s)",
        "memory": "Memory (MB)"
    }

    def organize_data(means: dict, stds: Optional[dict]):
        data = defaultdict(lambda: {"w2": [], "exec_time": [], "memory": []})
        for (dim, nlocs), vals in means.items():
            for k in ["w2", "exec_time", "memory"]:
                std_val = stds[(dim, nlocs)][k] if stds is not None else None
                data[nlocs][k].append((dim, vals[k], std_val))
        return data

    # Prepare primary dataset
    Q1 = organize_data(data_means, data_stds)
    dims1 = sorted({dim for (dim, _) in data_means.keys()})

    # Prepare comparison dataset if present
    has_comparison = data_means_comp is not None
    if has_comparison:
        Q2 = organize_data(data_means_comp, data_stds_comp)
        dims2 = sorted({dim for (dim, _) in data_means_comp.keys()})
    else:
        Q2 = None
        dims2 = None

    def plot_single_axis(ax, Q, dims, x_title, metric_key):
        unique_nlocs = sorted(Q.keys())
        cmap_colors = {
            nlocs: plt.get_cmap(COLORS[i])(0.65)
            for i, nlocs in enumerate(unique_nlocs)
        }

        for nlocs, metrics in Q.items():
            arr = sorted(metrics[metric_key], key=lambda x: x[0])
            dims_sorted, vals_mean, vals_std = zip(*arr)

            (line,) = ax.plot(
                dims_sorted,
                vals_mean,
                marker="o",
                color=cmap_colors[nlocs],
                label=_convert_sci_notation(nlocs)
            )

            # Fill std band if provided
            if any(s is not None for s in vals_std):
                lower = [m - (s or 0) for m, s in zip(vals_mean, vals_std)]
                upper = [m + (s or 0) for m, s in zip(vals_mean, vals_std)]
                ax.fill_between(dims_sorted, lower, upper, color=cmap_colors[nlocs], alpha=0.3)

        ax.set_xlabel(x_title)
        ax.grid(True)
        ax.set_xticks(dims)

    # Create figure
    ncols = 2 if has_comparison else 1

    for metric_key in metrics_to_plot:
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 5), squeeze=False)
        axes = axes[0]

        # Plot main dataset
        plot_single_axis(
            axes[0],
            Q1,
            dims1,
            x_axis_title,
            metric_key
        )
        axes[0].set_ylabel(metric_labels[metric_key])
        axes[0].legend(loc="best")

        # Plot comparison dataset if provided
        if has_comparison:
            plot_single_axis(
                axes[1],
                Q2,
                dims2,
                x_axis_title_comp or x_axis_title,
                metric_key
            )
            axes[1].legend(loc="best")
        plt.tight_layout()
        plt.show()

plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

COLORS = ['Blues', 'BuPu', 'PuRd', 'Greens', 'Oranges', 'Reds', 'Greys', 'Purples']

def _convert_sci_notation(n: int):
    x = torch.tensor(float(n))
    exp = int(torch.floor(torch.log10(x)))
    mantissa = float(x / (10 ** exp))

    if abs(mantissa - 1.0) < 1e-12:
        label = rf"$|\mathcal{{C}}| = 10^{{{exp}}}$"
    else:
        label = rf"$|\mathcal{{C}}| = {mantissa:g} \times 10^{{{exp}}}$"

    return label

if __name__ == '__main__':
    torch.manual_seed(10)

    ###################################################
    # Experiment: quantization and propagation        #
    ###################################################
    # Set parameters
    dimensions = [2, 10, 25, 50, 75, 100]
    nums_locs = [100, 1000, 10000]
    num_random_seeds = 10

    num_steps = 1

    # Collect data
    data_quant, data_prop = {}, {}
    for dimension, num_locs in product(dimensions, nums_locs):
        for random_seed in range(num_random_seeds):

            weight = sample_weight(dimension)  # random NN layer

            # Quantization
            distribution = MultivariateNormal(loc=torch.zeros(dimension), covariance_matrix=get_normalized_variance(dimension) * torch.eye(dimension))
            data_quant[(dimension, num_locs, random_seed)] = analyze_discretization(distribution=distribution, num_locs=num_locs)

            # Propagation
            dynamics = AdditiveNoiseDynamics(state_dynamics=NNLayerDynamics(weight=weight, bias=None))

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

    #######################################################
    # Experiment: probability concentration in a manifold #
    #######################################################

    dimension = 100
    nums_locs = [100, 1000, 10000]

    num_dists = 5
    small = 1e-5
    large = 1e-2

    distributions = manifold_distributions(dimension, num_dists=num_dists, small=small, large=large)

    # Collect data
    data_manifold = {}
    for distribution in distributions:
        manifold_dim = (distribution.covariance_matrix.diag() == small).sum().item()

        for num_locs in nums_locs:
            for random_seed in range(num_random_seeds):
                weight = sample_weight(dimension)  # random NN layer

                # Propagation
                dynamics = AdditiveNoiseDynamics(state_dynamics=NNLayerDynamics(weight=weight, bias=None))

                p = AmbiguityBall(
                    center=distribution,
                    radius=0.1
                )
                noise = AmbiguityBall(
                    center=MultivariateNormal(loc=torch.zeros(dimension),
                                              covariance_matrix=torch.eye(dimension) * 1e-4),
                    radius=0.01
                )

                data_manifold[(manifold_dim, num_locs, random_seed)] = analyze_propagation(dynamics=dynamics, p=p, noise=noise, num_locs=num_locs, num_steps=num_steps)


    means_prop_manifold, stds_prop_manifold = aggregate_stats(data_manifold)

    plot_dimension_analysis(
        data_means=means_prop_manifold,
        data_stds=stds_prop_manifold,
        x_axis_title=r"Dimension $d - d_{\emph{manifold}}$",
        metrics_to_plot= ["w2"],
        data_means_comp=means_prop,
        data_stds_comp= stds_prop,
        x_axis_title_comp= r"Dimension $d$"
    )

    plot_dimension_analysis(
        data_means=means_prop_manifold,
        data_stds=stds_prop_manifold,
        x_axis_title=r"Dimension $d - d_{\emph{manifold}}$",
        metrics_to_plot= ["exec_time"],
        data_means_comp=means_prop,
        data_stds_comp= stds_prop,
        x_axis_title_comp= r"Dimension $d$"
    )