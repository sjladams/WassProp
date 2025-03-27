import discretize_distributions as ds
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch
from scipy.stats import norm
import os

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
    X = torch.linspace(-3, 3, int(1e3)).unsqueeze(-1)
    Y = f(X)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(36, 12))

    # Gaussian density parameters
    mu = initial_dist.loc.item()
    sigma = np.sqrt(initial_dist.covariance_matrix.item())
    gaussian_density = norm.pdf(X.numpy(), loc=mu, scale=sigma)

    for i, (signature, bound) in enumerate(zip(signatures, bounds)):

        ax[i].plot(X, Y, label=r'$f(x)$')
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

        ax[i].text(mu-2.0, 0.5, rf'$h(\mathcal{{R}}, \mathcal{{C}}) = {bound:.2f}$',
                 ha='center', color='black',
                 bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

        ax[i].yaxis.set_visible(False)

    #ax[0].set_ylabel(r'$f(x)$')

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
    #plt.savefig(rf'C:\Users\efigueiredomot\Desktop\Papers\Wasserstein\sigmoid_signature_example.pdf', format='pdf')
    plt.show()

@torch.no_grad()
def plot_single_step(dynamics, w2_p1__q1_store: dict):
    w2_p__q_options = list(w2_p1__q1_store.keys())

    fig = plt.figure(figsize=(16, 16))
    for key in w2_p1__q1_store[w2_p__q_options[0]].keys():
        plt.plot(w2_p__q_options, [w2_p1__q1_store[w2_p__q][key] for w2_p__q in w2_p__q_options], label=key)

    plt.legend()
    plt.title(f"{dynamics.state_dynamics.__class__.__name__ if hasattr(dynamics, 'state_dynamics') else dynamics.__class__.__name__} (Lipschitz={dynamics.global_lipschitz:.2f})")
    plt.xlabel('$W_2(p,q)$')
    plt.xticks(w2_p__q_options)
    plt.ylabel(r'$W_2(f p, f \Delta q)$')
    plt.xlim(min(w2_p__q_options), max(w2_p__q_options))
    plt.show()


@torch.no_grad()
def plot_multi_step(dynamics, samples: dict, layout: str = "hist"):

    colors = plt.cm.tab10(np.linspace(0, 1, len(samples.keys())))

    time_steps = list(samples.keys())
    # time_steps = [time_steps[0], time_steps[-1]] # Get initial and final time steps

    if dynamics.num_state_dims >= 2:
        p_samples = torch.stack([samples[k]['p1_samples'] for k in samples.keys()])
        q_samples = torch.stack([samples[k]['q1_samples'] for k in samples.keys()])

        if layout == "scatter":
            fig, ax = plt.subplots(figsize=(12, 12))
            for k in time_steps:
                ax.scatter(p_samples[k,:,0], p_samples[k,:,1], label=rf'$t={k+1}$')
            ax.legend(loc='upper left')
            ax.set_xlim([-1., 1.])
            ax.set_ylim([-1., 1.])
        elif layout == "hist":
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
        else:
            raise NotImplementedError

        #plt.savefig(rf'C:\Users\efigueiredomot\Desktop\Papers\Wasserstein\{type}.pdf', format='pdf')
        plt.show()
    else:
        raise NotImplementedError


@torch.no_grad()
def plot_2d_ambiguity_balls(samples: dict, w2_p1__q1_store: dict, q_store, step_size: int = 1,
                            xlim: list = None, ylim: list = None, figsize: tuple = None, save_by: str = None):
    xlim = [-1, 1] if xlim is None else xlim
    ylim = [-1, 1] if ylim is None else ylim
    figsize = figsize if figsize is not None else (6 * (xlim[1] - xlim[0]), 6 * (ylim[1] - ylim[0]))

    time_steps = list(q_store.keys())[::step_size][:-1] # remove last time step, as compressed q not available
    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(time_steps) - 1)) for i in range(len(time_steps))]

    for tag in ['p1_samples', 'q1_samples']:
        fig, ax = plt.subplots(figsize=figsize)
        for k in time_steps:
            q = q_store[k+1]['q_compr']
            if isinstance(q, ds.MixtureMultivariateNormal):
                for i in range(q.num_components):
                    ambiguity_set = Circle(q.component_distribution.mean[i], w2_p1__q1_store[k]['w2_p1__q1_lagrangian_duality'], color=colors[k+1], fill=False, lw=2, alpha=0.5)
                    ax.add_patch(ambiguity_set)
            else:
                ambiguity_set = Circle(q.mean, w2_p1__q1_store[k]['w2_p1__q1_lagrangian_duality'], color=colors[k+1], fill=False, lw=2, alpha=0.5)
                ax.add_patch(ambiguity_set)

            ax.scatter(samples[k][tag][:, 0], samples[k][tag][:,1], color=colors[k+1], s=16, alpha=0.5)

        # ax.legend(loc='upper left')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        (plt.xlim(xlim), plt.xticks([])) if xlim is not None else None
        (plt.ylim(ylim), plt.yticks([])) if ylim is not None else None

        if save_by is not None:
            plt.savefig(f"{os.getcwd()}{os.sep}results{os.sep}{save_by}_path_{"true" if tag == "p1_samples" else "appr"}.pdf", format='pdf')
        else:
            plt.show()


@torch.no_grad()
def plot_2d_dynamics(f, xlim: list = None, ylim: list = None, scale: float = 1.0, figsize: tuple = None, save_by: str = None):
    xlim = [-1, 1] if xlim is None else xlim
    ylim = [-1, 1] if ylim is None else ylim
    figsize = figsize if figsize is not None else (6 * (xlim[1] - xlim[0]), 6 * (ylim[1] - ylim[0]))

    x = torch.linspace(xlim[0], xlim[1], 5 * int(xlim[1] - xlim[0]))
    y = torch.linspace(ylim[0], ylim[1], 5 * int(ylim[1] - ylim[0]))
    X, Y = torch.meshgrid(x, y, indexing="ij")
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)  # Shape (N, 2)

    with torch.no_grad():
        next_states = f.state_dynamics(grid_points)
        deltas = next_states - grid_points

    plt.figure(figsize=figsize)
    plt.quiver(
        grid_points[:, 0].numpy(),  # X coordinates
        grid_points[:, 1].numpy(),  # Y coordinates
        deltas[:, 0].numpy(),  # U: delta X
        deltas[:, 1].numpy(),  # V: delta Y
        angles='xy', scale_units='xy',
        scale=scale,
        width=0.003
    )
    (plt.xlim(xlim), plt.xticks([])) if xlim is not None else None
    (plt.ylim(ylim), plt.yticks([])) if ylim is not None else None

    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    if save_by is not None:
        plt.savefig(f"{os.getcwd()}{os.sep}results{os.sep}{save_by}_dynamics.pdf", format='pdf')
    else:
        plt.show()


