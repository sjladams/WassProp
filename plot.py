import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from analysis.configs import FOLDER

plt.style.use('seaborn-v0_8-bright')

plt.rcParams.update({
    'font.size': 40,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})


COLORS = ['Blues', 'BuPu', 'PuRd', 'Greens', 'Oranges', 'Reds', 'Greys', 'Purples',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

COLORS_HIST = [
    '#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30'
    ]

@torch.no_grad()
def plot_signatures(f, initial_dist, signatures, bounds):
    X = torch.linspace(-5, 8, int(1e3)).unsqueeze(-1)
    Y = f.state_dynamics(X)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(36, 12))

    # Gaussian density parameters
    mu = initial_dist.loc.item()
    sigma = np.sqrt(initial_dist.covariance_matrix.item())
    gaussian_density = norm.pdf(X.numpy(), loc=mu, scale=sigma)

    for i, (signature, bound) in enumerate(zip(signatures, bounds)):

        ax[i].plot(X, Y)
        ax[i].set_xlabel(r"$x$")

        ax[i].plot(X, gaussian_density, label=r'$\mathbb{P}$')

        # Discrete distribution setup
        arrow_x = signature.locs.squeeze().tolist()
        discrete_probs = signature.probs.tolist()

        # Add arrows representing discrete probabilities
        for xi, prob in zip(arrow_x, discrete_probs):
            ax[i].annotate('', xy=(xi, prob), xytext=(xi, 0),
                         arrowprops=dict(facecolor='green', edgecolor='green', width=0.8, headwidth=7),
                         )

        ax[i].text(mu, 0.4, rf'$h(\mathcal{{R}}, \mathcal{{C}}) = {bound:.2f}$',
                 ha='center', color='black',
                 bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

    ax[0].set_ylabel(r'$f(x)$')

    # Add a proxy artist for the annotation to include it in the legend
    annotation_label = r"$\Delta_{\mathcal{R}, \mathcal{C}}\#\mathbb{P}$"
    # Add the annotation to the legend
    handles, labels = plt.gca().get_legend_handles_labels()  # Get existing legend entries
    proxy_artist = plt.Line2D([0], [0], color="green")  # Create proxy
    handles.append(proxy_artist)
    labels.append(annotation_label)

    # Update the legend
    ax[0].legend(handles=handles, labels=labels, loc="upper left")
    plt.axhline(y=0, color="black", linewidth=0.2)

    plt.tight_layout()

    # Save the figure as EPS
    plt.savefig(rf'C:\Users\efigueiredomot\Desktop\Papers\Wasserstein\sigmoid.pdf', format='pdf')
    plt.show()

@torch.no_grad()
def plot_single_step(dynamics, w2_bounds: dict, **kwargs):
    w2_p__q_options = list(w2_bounds.keys())

    fig_w2_bounds = plt.figure()
    for key in w2_bounds[w2_p__q_options[0]].keys():
        if not key in ['sign_q']:
            plt.plot(w2_p__q_options, [w2_bounds[w2_p__q][key] for w2_p__q in w2_p__q_options], label=key)

    plt.legend()
    plt.title(f"{dynamics.state_dynamics.__class__.__name__ if hasattr(dynamics, 'state_dynamics') else dynamics.__class__.__name__} (Lipschitz={dynamics.global_lipschitz:.2f})")
    plt.xlabel('$W_2(p,q)$')
    plt.xticks(w2_p__q_options)
    plt.ylabel(r'$W_2(f p, f \Delta q)$')
    plt.xlim(min(w2_p__q_options), max(w2_p__q_options))
    plt.show()


@torch.no_grad()
def plot_multi_step(dynamics, samples: dict, type: str = "Undefined"):

    colors = plt.cm.tab10(np.linspace(0, 1, len(samples.keys())))

    time_steps = list(samples.keys())
    time_steps = [time_steps[0], time_steps[-1]] # Get initial and final time steps

    if dynamics.num_state_dims >= 2:
        p_samples = torch.stack([samples[k]['p'] for k in samples.keys()])
        q_samples = torch.stack([samples[k]['q'] for k in samples.keys()])

        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(24, 36))
        for k in time_steps:

            # Plot using hist2d with color intensity indicating the density
            ax[0][0].hist2d(p_samples[k,:,0], p_samples[k,:,1], bins=100, cmap=COLORS[k], alpha=0.8, cmin=0.1, label=rf'$t={k+1}$')
            ax[0][1].hist2d(q_samples[k,:,0], q_samples[k,:,1], bins=100, cmap=COLORS[k], alpha=0.8, cmin=0.1, label=rf'$t={k+1}$')

            # Plot only first dimension
            ax[1][0].hist(p_samples[k,:,0], color=colors[k], bins=50, density=True, label=rf'$t={k+1}$')
            ax[1][1].hist(q_samples[k, :, 0], color=colors[k], bins=50, density=True, label=rf'$t={k+1}$')

            # Plot only second dimension
            ax[2][0].hist(p_samples[k, :, 1], color=colors[k], bins=50, density=True, label=rf'$t={k+1}$')
            ax[2][1].hist(q_samples[k, :, 1], color=colors[k], bins=50, density=True, label=rf'$t={k+1}$')

        ax[0][0].set_title(r'$\mathbb{P}_{x_t}$ (actual distr.)')
        ax[0][1].set_title(r'$\hat{\mathbb{P}}_{x_t}$ (our approx.)')

        ax[0][0].set_xlabel(r'$x^{(0)}$')
        ax[0][1].set_xlabel(r'$x^{(0)}$')
        ax[0][0].set_ylabel(r'$x^{(1)}$')

        ax[1][0].set_xlabel(r'$x^{(0)}$')
        ax[1][1].set_xlabel(r'$x^{(0)}$')
        ax[1][0].set_ylabel('Frequency')

        ax[2][0].set_xlabel(r'$x^{(1)}$')
        ax[2][1].set_xlabel(r'$x^{(1)}$')
        ax[2][0].set_ylabel('Frequency')

        ax[0][0].grid(True)
        ax[0][1].grid(True)

        ax[0][0].axis('equal')
        ax[0][1].axis('equal')

        for ax in fig.axes:
            ax.legend(loc='upper left')

        plt.savefig(rf'C:\Users\efigueiredomot\Desktop\Papers\Wasserstein\{type}.pdf', format='pdf')
        plt.show()
    else:
        raise NotImplementedError

@torch.no_grad()
def plot_analysis(
        figure_name : str,
        dict_locs_1 : dict,
        dict_locs_2 : dict,
        axis_x : list,
        x_label : str = r"Number of locations ($|\mathcal{C}|$)",
        y_label : str = r"$\rho$-Wasserstein bound",
        log_scale : bool = True,
        different_y_axis : bool = False
    ):

    colors = plt.cm.tab10(np.linspace(0, 1, len(dict_locs_1)))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(28, 12), sharex='row', sharey='row')

    # Create the scatter plot
    for i, dynamics in enumerate(dict_locs_1.keys()):
        values_1 = dict_locs_1[dynamics]
        ax[0].scatter(axis_x, values_1, color=colors[i], label=dynamics)
        ax[0].plot(axis_x, values_1, color=colors[i], linestyle='--', linewidth=1)

        values_2 = dict_locs_2[dynamics]
        ax[1].scatter(axis_x, values_2, color=colors[i], label=dynamics)
        ax[1].plot(axis_x, values_2, color=colors[i], linestyle='--', linewidth=1)

    if log_scale:
        plt.xscale('log')
    ax[0].set_xlabel(x_label)
    ax[1].set_xlabel(x_label)
    ax[0].set_ylabel(y_label)
    if different_y_axis :
        ax[1].set_ylabel('Diff. betw. Rmk 1 and Thm 4')

    theta = [0, 0.1]
    # for i, a in enumerate(ax):
    #     a.text(
    #         0.95, 0.95,
    #         rf'$\theta = {theta[i]}$',
    #         horizontalalignment='right',
    #         verticalalignment='top',
    #         transform=a.transAxes,
    #         bbox = dict(
    #             facecolor='white',
    #             edgecolor='black',
    #             boxstyle='round,pad=0.3'
    #         )
    #     )

    legend = ax[1].legend(
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        ncol=1,
        frameon=True
    )
    ax[0].grid(True, alpha=0.7)
    ax[1].grid(True, alpha=0.7)

    # Show the plot
    plt.tight_layout()
    file = os.path.join(FOLDER, figure_name)
    plt.savefig(file, format='pdf')
    plt.show()

@torch.no_grad()
def plot_optimize_locs(
        figure_name : str,
        dict_lipschitz : dict,
        dict_duality : dict,
        dict_optimize : dict,
        dict_random : dict,
        axis_x : list,
        label_names : list,
        x_label : str = r'$\rho$-Wasserstein ball radius $\theta$',
        y_label : str = r"$\mathbb{W}_{\rho}(f \# \mathbb{P}, f \# \Delta_{\mathcal{R}, \mathcal{C}} \# \mathbb{P})$",
        log_scale : bool = True
    ):

    colors = plt.cm.tab10(np.linspace(0, 1, len(dict_lipschitz)))

    # Create the scatter plot
    plt.figure(figsize=(18, 12))
    for i, (key, values) in enumerate(dict_lipschitz.items()):
        plt.scatter(axis_x, values, color='red', label='Global Lipschitz')
        plt.plot(axis_x, values, color='red', linestyle='--', linewidth=1)

        plt.scatter(axis_x, dict_duality[key], color='green', label='Lagrangian duality (no locs optim.)')
        plt.plot(axis_x, dict_duality[key], color='green', linestyle='--', linewidth=1)

        plt.scatter(axis_x, dict_optimize[key], color='blue', label='Lagrangian duality (locs optim.)')
        plt.plot(axis_x, dict_optimize[key], color='blue', linestyle='--', linewidth=1)

        plt.scatter(axis_x, dict_random[key], color='orange', label='Random locs')
        plt.plot(axis_x, dict_random[key], color='orange', linestyle='--', linewidth=1)

    if log_scale:
        plt.xscale('log')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    file = os.path.join(FOLDER, figure_name)
    plt.savefig(file, format='pdf')
    plt.show()