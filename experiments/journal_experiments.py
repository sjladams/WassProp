import os
import copy
import json
import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch
import discretize_distributions as dd
import numpy as np
from scipy.stats import norm

from wass_prop import single_step, multi_step, multi_step_empirical, single_step_empirical, SampledPath, AmbiguityBall
import wass_prop.wasserstein as wasserstein
from wass_prop.propagation import multi_step_distribution
from wass_prop.utils_distributions import w2_discrete

from dynamics import get_stoch_dynamics
from handlers import parse_arguments
import utils


# Systems shared by the convergence-analysis figures: name -> (dynamics_type, dynamics_setting, default_num_locs, color).
# default_num_locs is only used where a fixed number of locations is needed (e.g. w2_p__q_convergence_analysis).
SYSTEMS = {
    'Sigmoid (1D)': ('SigmoidDynamics', 0, 100, 'tab:blue'),
    'Bounded Linear (2D)': ('BoundedLinearDynamics', 0, 100, 'tab:green'),
    'Mountain Car (2D)': ('MountainCarJournalDynamics', 0, 100, 'tab:olive'),
    'Dubins Car (3D)': ('DubinsCarDynamics', 0, 1000, 'tab:cyan'),
    'Quadruple-Tank (4D)': ('LinearDynamics', 0, 1000, 'tab:purple'),
    'NN Layer (10D)': ('DiagonalSigmoidDynamics', 2, 1000, 'tab:pink'),
}

plt.style.use('seaborn-v0_8-bright')
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def sigmoid_example():
    args = parse_arguments(
        dynamics_type='SigmoidDynamics',
        dynamics_setting=0,
    )

    dynamics = get_stoch_dynamics(name=args.dynamics_type, **vars(args.dynamics))
    initial_dist = utils.get_initial_dist(loc=args.initial_dist.loc, variance=args.initial_dist.variance)

    signatures, bounds = [], []
    for num_locs in [5, 10, 100]:
        scheme = dd.generate_scheme(dist=initial_dist, scheme_size=num_locs)
        signature, _ = dd.discretize(initial_dist, scheme)
        signatures.append(signature)

        fn_sq_w2_f_q__f_disc_q = wasserstein.get_fn_sq_w2_f_q__f_disc_q(initial_dist, signature, dynamics.state_dynamics)
        bound = fn_sq_w2_f_q__f_disc_q().sqrt()
        bounds.append(bound)

    X = torch.linspace(-3, 3, int(1e3)).unsqueeze(-1)
    Y = dynamics.state_dynamics(X)

    mu = initial_dist.loc.item()
    sigma = np.sqrt(initial_dist.covariance_matrix.item())
    gaussian_density = norm.pdf(X.numpy(), loc=mu, scale=sigma)

    with plt.rc_context({'font.size': 55}):
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(36, 12), constrained_layout=True)

        for i, (signature, bound) in enumerate(zip(signatures, bounds)):
            ax[i].plot(X, Y, label=r'$f(x)$', linewidth=3.5)
            ax[i].plot(X, gaussian_density, label=r'$\mathbb{P}$', linewidth=3.5)
            ax[i].set_xlabel(r"$x$")

            arrow_x = signature.locs.squeeze().tolist()
            discrete_probs = signature.probs.tolist()
            for xi, prob in zip(arrow_x, discrete_probs):
                ax[i].arrow(xi, 0, 0, prob, length_includes_head=False,
                          width=0.017, head_width=0.085, head_length=0.03,
                          facecolor='green', edgecolor='green')

            ax[i].text(0.95, 0.95, rf'$\mathcal{{W}}_{{\mathcal{{R}}, \mathcal{{C}}}} = {bound:.2f}$',
                     transform=ax[i].transAxes,
                     ha='right', va='top', color='black',
                     bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.8, boxstyle='round,pad=0.3'))

            ax[i].yaxis.set_visible(False)

        # proxy artist for the discretized pushforward, which isn't drawn via a labeled plot call
        handles, labels = ax[0].get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color="green", linewidth=3.5))
        labels.append(r"$\Delta_{\mathcal{R}, \mathcal{C}}\#\mathbb{P}$")
        ax[0].legend(handles=handles, labels=labels, loc="upper left")

        plt.savefig(os.path.join(RESULTS_DIR, 'sigmoid_signature_example.pdf'), bbox_inches='tight')
        plt.show()

def _build_system(dynamics_type, setting, num_locs, num_samples=1000):
    args = parse_arguments(
        dynamics_type=dynamics_type,
        dynamics_setting=setting,
        num_locs=num_locs,
        num_samples=num_samples,
    )
    dynamics = get_stoch_dynamics(name=args.dynamics_type, **vars(args.dynamics))
    initial_dist = utils.get_initial_dist(loc=args.initial_dist.loc, variance=args.initial_dist.variance)
    noise_dist = utils.get_noise_dist(loc=args.noise_dist.loc, variance=args.noise_dist.variance)
    return args, dynamics, initial_dist, noise_dist

def convergence_analysis(dynamics_type, setting, num_locs, w2_p__q):
    args, dynamics, initial_dist, noise_dist = _build_system(dynamics_type, setting, num_locs)

    results = dict()
    for method in ['global_lipschitz', 'lagrangian_duality']:
        w2 = single_step(
            dynamics=dynamics,
            q=AmbiguityBall(initial_dist, w2_p__q),
            noise=AmbiguityBall(noise_dist, 0.),
            num_locs=args.num_locs,
            use_lagrangian_duality=method=='lagrangian_duality'
        ).w2
        results[method] = float(w2)
    return results


def num_loc_convergence_analysis(w2_p__q_values: list[float]):
    num_locs_experiment = [10, 100, 1000, 10000]

    fig, axes = plt.subplots(nrows=1, ncols=len(w2_p__q_values), figsize=(7.5, 3.5),
                              sharey=True, constrained_layout=True)
    if len(w2_p__q_values) == 1:
        axes = [axes]

    for ax, w2_p__q in zip(axes, w2_p__q_values):
        for dynamics_name, (dynamics_type, setting, _, color) in SYSTEMS.items():
            w2_values = [convergence_analysis(dynamics_type, setting, num_locs, w2_p__q)['lagrangian_duality']
                         for num_locs in num_locs_experiment]

            ax.plot(num_locs_experiment, w2_values, marker='o', linestyle='--', label=dynamics_name,
                    color=color)

        ax.set_xscale('log')
        ax.set_ylim(bottom=0, top=1.55)
        ax.set_xlabel(r'Number of locations ($|\mathcal{C}|$)')
        ax.grid(True)
        ax.text(0.95, 0.95, rf'$\theta = {w2_p__q}$',
                transform=ax.transAxes,
                ha='right', va='top', color='black',
                bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.8, boxstyle='round,pad=0.3'))

    handles, labels = axes[0].get_legend_handles_labels()
    handles = [copy.copy(h) for h in handles]
    for h in handles:
        h.set_linestyle('None')

    axes[0].set_ylabel('2-Wasserstein bound')
    fig.legend(handles, labels, loc='outside center right')
    plt.savefig(os.path.join(RESULTS_DIR, 'increase_num_locs_analysis.pdf'), bbox_inches='tight')
    plt.show()


def w2_p__q_convergence_analysis():
    w2_p__q_options = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    results = {}
    for dynamics_name, (dynamics_type, setting, num_locs, _) in SYSTEMS.items():
        results[dynamics_name] = {}
        for w2_p__q in w2_p__q_options:
            results[dynamics_name][w2_p__q] = convergence_analysis(dynamics_type, setting, num_locs, w2_p__q)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 3.5), 
                             sharey=True, constrained_layout=True)

    for dynamics_name, data in results.items():
        w2_values = np.array([data[w2]['lagrangian_duality'] for w2 in w2_p__q_options])
        w2_values_global = np.array([data[w2]['global_lipschitz'] for w2 in w2_p__q_options])
        color = SYSTEMS[dynamics_name][3]

        axes[0].plot(w2_p__q_options, w2_values, marker='o', linestyle='--', label=dynamics_name, color=color)
        axes[1].plot(w2_p__q_options, w2_values_global - w2_values, marker='o', linestyle='--',
                      label=dynamics_name, color=color)

    for ax in axes:
        ax.set_xlabel(r'2-Wasserstein ball radius $\theta$')
        ax.grid(True)

    axes[0].set_ylabel('2-Wasserstein bound')
    axes[1].set_ylabel('Global Lip. $-$ Lin. Coeff.')

    handles, labels = axes[0].get_legend_handles_labels()
    handles = [copy.copy(h) for h in handles]
    for h in handles:
        h.set_linestyle('None')

    fig.legend(handles, labels, loc='outside center right')
    plt.savefig(os.path.join(RESULTS_DIR, 'wass_ball_radius_analysis.pdf'), bbox_inches='tight')
    plt.show()


def _w2_series_to_reference(true_samples, get_comparison, ordered_indices):
    result = {-1: 0.}
    for k in set(ordered_indices) - {-1}:
        w2 = w2_discrete(true_samples.at(k), get_comparison(k)).item()
        result[k] = round(float(w2), 4)
    return result


_TIME_KEY = 0

@contextmanager
def _timer(store: dict, key: int = 0, num_steps: int = 1):
    """Times the wrapped block and writes {key: round(elapsed / num_steps, 4)} into `store`."""
    start = time.perf_counter()
    yield
    store[key] = round((time.perf_counter() - start) / num_steps, 4)


def _aggregate_repeated_series(series_list: list[dict], std_multiplier: float = 1.0) -> dict:
    """Combine several {time_step: value} series (one per repeat) into
    {time_step: {'mean': ..., 'std': ...}}, where 'std' is std_multiplier * sample std."""
    keys = series_list[0].keys()
    result = {}
    for k in keys:
        values = np.array([series[k] for series in series_list])
        result[k] = {
            'mean': round(float(values.mean()), 4),
            'std': round(float(values.std(ddof=1) * std_multiplier), 4) if len(values) > 1 else 0.,
        }
    return result


def _save_json(data: dict, filepath: str) -> None:
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=1)


def _load_results_json(filepath: str) -> dict:
    """Load a {dynamics_name: {method: {time_step: value}}} results/timings JSON, converting
    the (JSON-stringified) time_step keys back to int."""
    with open(filepath) as f:
        data = json.load(f)
    return {
        name: {method: {int(k): v for k, v in series.items()} for method, series in methods.items()}
        for name, methods in data.items()
    }


def results_to_latex_table(results: dict, filepath: str, column_order: list | None = None) -> str:
    """Render a {dynamics_name: {method: {time_step: value}}} results dict as a LaTeX
    table with time steps as rows and a multicolumn block per dynamics, subdivided into
    one column per method.

    column_order: if given, methods are ordered to match it (per dynamics), with any
        methods not listed appended at the end in their original order. If None, methods
        keep their dict order."""
    dynamics_names = list(results.keys())
    if column_order is None:
        methods_per_dynamics = {name: list(methods.keys()) for name, methods in results.items()}
    else:
        methods_per_dynamics = {
            name: [m for m in column_order if m in methods] + [m for m in methods if m not in column_order]
            for name, methods in results.items()
        }

    time_steps = sorted({
        k for methods in results.values() for series in methods.values() for k in series.keys()
    })

    col_spec = 'c' + ''.join('c' * len(methods_per_dynamics[name]) for name in dynamics_names)

    lines = [
        r'\begin{tabular}{' + col_spec + '}',
        r'\toprule',
    ]

    # header row 1: dynamics name spanning its methods
    header1_cells = ['']
    cmidrules = []
    col_start = 2
    for name in dynamics_names:
        n_methods = len(methods_per_dynamics[name])
        header1_cells.append(rf'\multicolumn{{{n_methods}}}{{c}}{{{name}}}')
        cmidrules.append(rf'\cmidrule(lr){{{col_start}-{col_start + n_methods - 1}}}')
        col_start += n_methods
    lines.append(' & '.join(header1_cells) + r' \\')
    lines.extend(cmidrules)

    # header row 2: method labels
    header2_cells = [r'$t$']
    for name in dynamics_names:
        for method in methods_per_dynamics[name]:
            header2_cells.append(method)
    lines.append(' & '.join(header2_cells) + r' \\')
    lines.append(r'\midrule')

    # data rows
    for k in time_steps:
        row_cells = [str(k + 1)]
        for name in dynamics_names:
            for method in methods_per_dynamics[name]:
                value = results[name][method].get(k)
                if value is None:
                    cell = '--'
                elif isinstance(value, dict):
                    cell = rf"{value['mean']:.4f} $\pm$ {value['std']:.4f}"
                else:
                    cell = f'{value:.4f}'
                row_cells.append(cell)
        lines.append(' & '.join(row_cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    table_str = '\n'.join(lines)
    with open(filepath, 'w') as f:
        f.write(table_str)
    return table_str


def dynamic_system_analysis(num_repeats: int = 10, std_multiplier: float = 1.0, num_samples: int = 100000):
    """
    num_repeats: number of times to redraw samples for the stochastic methods
        ('empirical', 'mc', 'sigma'); when > 1, their reported value per time step
        becomes {'mean': ..., 'std': std_multiplier * sample_std}.
    std_multiplier: e.g. 3. for a 3-sigma band; standard reporting practice is 1
        (a single sample std, or std/sqrt(num_repeats) for the standard error of the mean).
    """
    systems = {
        # 'Sigmoid (1D)': ('SigmoidDynamics', 0, 100, 'tab:blue'),
        # 'Bounded Linear (2D)': ('BoundedLinearDynamics', 0, 100, 'tab:green'),
        'NN Layer (3D)': ('DiagonalSigmoidDynamics', 3, 100, 'tab:pink'),
        'Mountain Car (2D)': ('MountainCarJournalDynamics', 0, 100, 'tab:olive'),
        # 'Mountain Car (2D)': ('MountainCarDynamics', 0, 100, 'tab:olive'),
        # 'Dubins Car (3D)': ('DubinsCarDynamics', 0, 1000, 'tab:cyan'),
        'Quadruple-Tank (4D)': ('LinearDynamics', 1, 100, 'tab:purple'),
        # 'NN Layer (10D)': ('DiagonalSigmoidDynamics', 2, 1000, 'tab:pink'),
    }

    num_time_steps = 50

    results = dict()
    timings = dict()
    num_systems = len(systems)
    for system_idx, (dynamics_name, (dynamics_type, setting, num_locs, _)) in enumerate(systems.items(), start=1):
        print(f"\n=== [{system_idx}/{num_systems}] {dynamics_name} ({dynamics_type}) ===")
        args, dynamics, initial_dist, noise_dist = _build_system(
            dynamics_type, setting, num_locs, num_samples=num_samples
        )
        q = AmbiguityBall(initial_dist, 0.)
        noise = AmbiguityBall(noise_dist, 0.)

        results[dynamics_name] = dict()
        timings[dynamics_name] = dict()
        for method in ['global_lipschitz', 'lagrangian_duality']:
            print(f"  [{dynamics_name}] method={method}: propagating {num_time_steps} steps...")
            timings[dynamics_name][method] = {}
            with _timer(timings[dynamics_name][method], num_steps=num_time_steps):
                path = multi_step(
                    dynamics=dynamics,
                    q=q,
                    noise=noise,
                    num_time_steps=num_time_steps,
                    use_lagrangian_duality=method == 'lagrangian_duality',
                    num_locs=args.num_locs,
                    use_additive_noise=False,
                )
            results[dynamics_name][method] = {k: round(float(path.at(k).w2), 4) for k in path.ordered_indices}
            final_w2 = results[dynamics_name][method][path.ordered_indices[-1]]
            elapsed = timings[dynamics_name][method][_TIME_KEY]
            print(f"  [{dynamics_name}] method={method}: done in {elapsed:.2f}s, final w2={final_w2}")

        print(f"  [{dynamics_name}] sigma-point: propagating {num_time_steps} steps...")
        sigma_path_timing = {}
        with _timer(sigma_path_timing):
            sigma_path = multi_step_distribution(
                dynamics=dynamics,
                q=initial_dist,
                noise=noise_dist,
                num_time_steps=num_time_steps,
                num_locs=100,
                use_additive_noise=False,
                configuration='cross',
            )
        sigma_path_time = sigma_path_timing[_TIME_KEY]
        print(f"  [{dynamics_name}] sigma-point: done in {sigma_path_time:.2f}s")

        empirical_series, mc_series, sigma_series = [], [], []
        empirical_times, mc_times, sigma_times = [], [], []
        for i in range(num_repeats):
            print(f"  [{dynamics_name}] repeat {i + 1}/{num_repeats}")
            print(f"    empirical: sampling {args.num_samples} trajectories over {num_time_steps} steps...")
            empirical_time = {}
            with _timer(empirical_time, num_steps=num_time_steps):
                true_samples = multi_step_empirical(
                    dynamics=dynamics,
                    p_emp=initial_dist.sample((args.num_samples,)),
                    noise=noise,
                    num_time_steps=num_time_steps,
                ).detach()
            empirical_times.append(empirical_time)
            print(f"    empirical: done in {empirical_time[_TIME_KEY]:.2f}s")

            empirical_series.append(_w2_series_to_reference(
                true_samples, lambda k: path.at(k).center, path.ordered_indices
            ))

            print(f"    mc: sampling {args.num_samples} trajectories over {num_time_steps} steps...")
            mc_time = {}
            with _timer(mc_time, num_steps=num_time_steps):
                mc_samples = multi_step_empirical(
                    dynamics=dynamics,
                    p_emp=q.sample(args.num_samples),
                    noise=noise,
                    num_time_steps=num_time_steps,
                ).detach()
            mc_times.append(mc_time)
            print(f"    mc: done in {mc_time[_TIME_KEY]:.2f}s")
            mc_series.append(_w2_series_to_reference(
                true_samples, mc_samples.at, path.ordered_indices
            ))

            print(f"    sigma: sampling {args.num_samples} points from precomputed sigma path...")
            sigma_time = {}
            with _timer(sigma_time):
                sigma_samples = SampledPath(
                    {k: sigma_path.at(k).sample((args.num_samples,)) for k in path.ordered_indices}
                ).detach()
            sigma_time[_TIME_KEY] = round((sigma_time[_TIME_KEY] + sigma_path_time) / num_time_steps, 4)
            sigma_times.append(sigma_time)
            print(f"    sigma: done in {sigma_time[_TIME_KEY]:.2f}s")
            sigma_series.append(_w2_series_to_reference(
                true_samples, sigma_samples.at, path.ordered_indices
            ))

        if num_repeats == 1:
            results[dynamics_name]['empirical'] = empirical_series[0]
            results[dynamics_name]['mc'] = mc_series[0]
            results[dynamics_name]['sigma'] = sigma_series[0]
            timings[dynamics_name]['empirical'] = empirical_times[0]
            timings[dynamics_name]['mc'] = mc_times[0]
            timings[dynamics_name]['sigma'] = sigma_times[0]
        else:
            results[dynamics_name]['empirical'] = _aggregate_repeated_series(empirical_series, std_multiplier)
            results[dynamics_name]['mc'] = _aggregate_repeated_series(mc_series, std_multiplier)
            results[dynamics_name]['sigma'] = _aggregate_repeated_series(sigma_series, std_multiplier)
            timings[dynamics_name]['empirical'] = _aggregate_repeated_series(empirical_times, std_multiplier)
            timings[dynamics_name]['mc'] = _aggregate_repeated_series(mc_times, std_multiplier)
            timings[dynamics_name]['sigma'] = _aggregate_repeated_series(sigma_times, std_multiplier)

    results_path = os.path.join(RESULTS_DIR, 'dynamic_system_analysis.json')
    timings_path = os.path.join(RESULTS_DIR, 'dynamic_system_analysis_timings.json')
    _save_json(results, results_path)
    _save_json(timings, timings_path)
    print(f"\nSaved results to {results_path}")
    print(f"Saved timings to {timings_path}")

    render_dynamic_system_analysis_tables()


def render_dynamic_system_analysis_tables(column_order: list | None = None):
    """Render the .tex tables from the JSON files saved by dynamic_system_analysis, without
    re-running the (expensive) computation. Re-run this on its own after editing the table
    layout (e.g. column_order)."""
    if column_order is None:
        column_order = ['mc', 'sigma', 'empirical', 'global_lipschitz', 'lagrangian_duality']

    results = _load_results_json(os.path.join(RESULTS_DIR, 'dynamic_system_analysis.json'))
    timings = _load_results_json(os.path.join(RESULTS_DIR, 'dynamic_system_analysis_timings.json'))

    results_to_latex_table(
        results, os.path.join(RESULTS_DIR, 'dynamic_system_analysis_table.tex'),
        column_order=column_order,
    )
    results_to_latex_table(
        timings, os.path.join(RESULTS_DIR, 'dynamic_system_analysis_timings_table.tex'),
        column_order=column_order,
    )


def uniform_vs_optimized():
    num_locs_experiment = [5, 10, 100, 1000]
    setting_per_dimension = {1:1, 2:2, 3:3, 4:4}

    results = dict()
    for num_dims, setting in setting_per_dimension.items():
        results[num_dims] = dict()

        for num_locs in num_locs_experiment:
            results[num_dims][num_locs] = dict()
        
            args = parse_arguments(
                dynamics_type="DiagonalLinearBoundedDynamics",
                dynamics_setting=setting,
                num_locs=num_locs,
            )

            dynamics = get_stoch_dynamics(name=args.dynamics_type, **vars(args.dynamics))
            q = utils.get_initial_dist(loc=args.initial_dist.loc, variance=args.initial_dist.variance)

            # Optimized
            scheme_q = dd.generate_scheme(dist=q, scheme_size=num_locs, configuration='grid')
            unif_scheme_q = dd.generate_scheme(dist=q, scheme_size=num_locs, configuration='uniform_grid')

            for tag, scheme in zip(['optimized', 'uniform'], [scheme_q, unif_scheme_q]):
                disc_q, _ = dd.discretize(q, scheme)

                w2 = wasserstein.compute_w2_f_q__f_disc_q_lagrangian_duality(
                    q=q,
                    disc_q=disc_q, 
                    f=dynamics.state_dynamics, 
                )

                results[num_dims][num_locs][tag] = round(w2.item(), 4)

    print(json.dumps(results, indent=1))


def mountain_car_mc_plot():
    args, dynamics, initial_dist, noise_dist = _build_system(
        'MountainCarJournalDynamics', setting=0, num_locs=100, num_samples=10000
    )

    num_time_steps = 10

    q = AmbiguityBall(initial_dist, 0.0)
    noise = AmbiguityBall(noise_dist, 0.0)

    path = multi_step(
        dynamics=dynamics, 
        q=q, 
        noise=noise,
        num_time_steps=num_time_steps,
        use_lagrangian_duality=True,
        use_additive_noise=False,
        num_locs=args.num_locs,
    )

    true_samples = multi_step_empirical(
        dynamics=dynamics,
        p_emp=q.sample(args.num_samples),
        noise=noise,
        num_time_steps=num_time_steps,
    ).detach()
    approx_samples = SampledPath({k: path.at(k).sample(args.num_samples) for k in path.ordered_indices}).detach()


    # with plt.rc_context({'font.size': 40}):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 36))
    cmap = cm.get_cmap("managua")
    colors = cmap(np.linspace(0,1,num_time_steps+1))
    for k, color in zip(true_samples.ordered_indices, colors):
        # Plot using hist2d with color intensity indicating the density
        mono_cmap = mcolors.LinearSegmentedColormap.from_list("mono", ["white", color])
        ax[0,0].hist2d(true_samples.at(k)[:,0], true_samples.at(k)[:,1], bins=100, cmap=mono_cmap, alpha=0.8, cmin=0.1, label=rf'$t={k+1}$')
        ax[0,1].hist2d(approx_samples.at(k)[:,0], approx_samples.at(k)[:,1], bins=100, cmap=mono_cmap, alpha=0.8, cmin=0.1, label=rf'$t={k+1}$')

    for k, color in zip([0,num_time_steps-1], ['tab:blue', 'tab:cyan']):
        # Plot only first dimension
        ax[1,0].hist(true_samples.at(k)[:,0], color=color, bins=50, density=True, label=rf'$t={k+1}$')
        ax[1,1].hist(approx_samples.at(k)[:,0], color=color, bins=50, density=True, label=rf'$t={k+1}$')

        # Plot only second dimension
        ax[2,0].hist(true_samples.at(k)[:, 1], color=color, bins=50, density=True, label=rf'$t={k+1}$')
        ax[2,1].hist(approx_samples.at(k)[:, 1], color=color, bins=50, density=True, label=rf'$t={k+1}$')

    ax[0,0].set_title(r'$\mathbb{P}_{x_t}$ (actual distr.)')
    ax[0,1].set_title(r'$\hat{\mathbb{P}}_{x_t}$ (our approx.)')

    ax[0,0].set_xlabel(r'$x^{(0)}$')
    ax[0,1].set_xlabel(r'$x^{(0)}$')
    ax[0,0].set_ylabel(r'$x^{(1)}$')

    ax[1,0].set_xlabel(r'$x^{(0)}$')
    ax[1,1].set_xlabel(r'$x^{(0)}$')
    ax[1,0].set_ylabel('Frequency')

    ax[2,0].set_xlabel(r'$x^{(1)}$')
    ax[2,1].set_xlabel(r'$x^{(1)}$')
    ax[2,0].set_ylabel('Frequency')

    ax[0,0].grid(True)
    ax[0,1].grid(True)

    ax[0,0].axis('equal')
    ax[0,1].axis('equal')

    for ax in fig.axes:
        ax.legend(loc='upper left')

    plt.show()
    plt.savefig(os.path.join(RESULTS_DIR, 'multistep_mountain_car.pdf'), bbox_inches='tight')


def run_single_step_no_ambiguity(dynamics, q, noise, num_locs, num_samples_empirical=50000):
    results = dict()
    for method in ['lagrangian_duality']:
        propagated_ball = single_step(
            dynamics=dynamics,
            q=q,
            noise=noise,
            num_locs=num_locs,
            use_lagrangian_duality=True if method == 'lagrangian_duality' else False,
        )
        results[method] = round(float(propagated_ball.w2), 4)

    p_samples = q.sample(num_samples_empirical) # sample from zero radius set is equivalent to sampling from center
    samples_empirical = single_step_empirical(
        dynamics=dynamics,
        p_emp=p_samples,
        noise=noise,
        num_samples=num_samples_empirical,
    )

    empirical_w2 = w2_discrete(samples_empirical, propagated_ball.center).item()
    results['empirical'] = empirical_w2

    return results


@torch.no_grad()
def conservativeness_analysis():
    systems = SYSTEMS
    systems['Sigmoid (1D)'] = ('SigmoidDynamics', 0, 10, 'tab:blue')
    systems['NN Layer (10D)'] = ('DiagonalSigmoidDynamics', 2, 10000, 'tab:pink')

    results = {}
    for dynamics_name, (dynamics_type, setting, num_locs, _) in systems.items():
        args = parse_arguments(
            dynamics_type=dynamics_type,
            dynamics_setting=setting,
            num_locs=num_locs,
        )

        dynamics = get_stoch_dynamics(name=args.dynamics_type, **vars(args.dynamics))
        initial_dist = utils.get_initial_dist(loc=args.initial_dist.loc, variance=args.initial_dist.variance)
        noise_dist = dd.CategoricalFloat(
            locs=torch.zeros(1, len(args.initial_dist.loc)),
            probs=torch.tensor([1.0])
        )

        q = AmbiguityBall(initial_dist, 0.)
        noise = AmbiguityBall(noise_dist, 0.) # we don't consider the noise in this experiment (Dirac at zero)

        results[dynamics_name] = {}
        results[dynamics_name][num_locs] = run_single_step_no_ambiguity(dynamics, q, noise, num_locs)

    print(json.dumps(results, indent=1))


if __name__ == '__main__':
    torch.manual_seed(0)

    # # Figure 3
    # sigmoid_example()

    # # Figure 5
    # num_loc_convergence_analysis(w2_p__q_values=[0, 0.1])

    # # Figure 6
    # w2_p__q_convergence_analysis()

    # # Figure 7
    # mountain_car_mc_plot()

    # # Figure 8
    # # See dimensional_analysis.py

    # # Table I
    # uniform_vs_optimized()

    # Table II
    dynamic_system_analysis()

    # # Table III
    # conservativeness_analysis()