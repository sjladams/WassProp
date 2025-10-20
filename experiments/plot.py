import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch
from scipy.stats import norm
from typing import Union, Optional, List, Dict, Tuple
from matplotlib.ticker import MaxNLocator

from duq_via_wasserstein import SampledPath, Path

plt.style.use('seaborn-v0_8-bright')

# plt.rcParams.update({
#     'font.size': 40,
#     'text.usetex': True,
#     'text.latex.preamble': r'\usepackage{amsfonts}'
# })


COLORS = ['Blues', 'BuPu', 'PuRd', 'Greens', 'Oranges', 'Reds', 'Greys', 'Purples',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

COLORS_HIST = [
    '#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30'
    ]

@torch.no_grad()
def plot_signatures(f, initial_dist, signatures, bounds):  # TODO to be updated
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
def plot_2d_ambiguity_balls(
    samples: SampledPath, 
    path: Optional[Path] = None,
    patch_creator = None, text_creator = None,
    step_size: int = 1,
    xlim: Optional[List] = None, ylim: Optional[List] = None, figsize: Optional[tuple] = None, 
    save_by: str = 'path', save: bool = False, 
    title: Optional[str] = None
):
    xlim = [-1, 1] if xlim is None else xlim
    ylim = [-1, 1] if ylim is None else ylim
    figsize = figsize if figsize is not None else (6 * (xlim[1] - xlim[0]), 6 * (ylim[1] - ylim[0]))

    fig, ax = plt.subplots(figsize=figsize)

    if patch_creator is not None:
        for patch in patch_creator():
            plt.gca().add_patch(patch)
    if text_creator is not None:
        for text in text_creator():
            plt.text(**text)

    time_steps = samples.ordered_indices[::step_size][:-1]
    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(time_steps) - 1)) for i in range(len(time_steps))]

    for k in time_steps:
        ax.scatter(samples.at(k)[:, 0], samples.at(k)[:, 1], color=colors[k + 1], s=16, alpha=0.5)
        if path is not None:
            amb_ball = path.at(k)
            ambiguity_set = Circle(amb_ball.center.mean, amb_ball.w2, color=colors[k+1], fill=False, lw=2, alpha=1.0)
            ax.add_patch(ambiguity_set)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    plt.xlim(xlim) if xlim is not None else None
    plt.ylim(ylim) if ylim is not None else None
    plt.tight_layout()

    if title is not None:
        plt.title(title)

    if save:
        plt.savefig(f"{save_by}.pdf", format='pdf')
    else:
        plt.show()


@torch.no_grad()
def plot_2d_dynamics(
        f, patch_creator = None, text_creator = None,
        xlim: Optional[List] = None, ylim: Optional[List] = None, scale: Optional[float] = 1.0, figsize: Optional[Tuple] = None, 
        save_by: str = 'dynamics', save: bool = False, title: Optional[str] = None
):

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
    if patch_creator is not None:
        for patch in patch_creator():
            plt.gca().add_patch(patch)
    if text_creator is not None:
        for text in text_creator():
            plt.text(**text)

    plt.xlim(xlim) if xlim is not None else None
    plt.ylim(ylim) if ylim is not None else None

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    if save:
        plt.savefig(f"{save_by}.pdf", format='pdf')
    else:
        plt.show()


# ----- Plotting Utilities from the discretize_distributions package ---------------------------------------------------

import discretize_distributions.cell as dd_cell
import discretize_distributions.schemes as dd_schemes


def plot_2d_dist(ax, dist, num_samples=10000):
    samples = dist.sample((num_samples,))
    ax.hist2d(samples[:,0], samples[:,1], bins=[50,50], density=True)
    return ax

def plot_2d_cat_float(ax, dist, s: float = 500, c: str = 'red', **kwargs):
    ax.scatter(
        dist.locs[:, 0],
        dist.locs[:, 1],
        s=dist.probs * s,
        c=c,
        **kwargs
    )
    return ax

def plot_2d_grid(ax, grid, s: float = 10, c: str = 'red', **kwargs):
    ax.scatter(
        grid.points[:, 0],
        grid.points[:, 1],
        s=s,
        c=c,
        **kwargs
    )
    return ax

def plot_2d_cell(ax, cell: dd_cell.Cell, c: str = 'blue', linewidth: float = 2, **kwargs):
    verts = sort_vertices_counterclockwise(cell.vertices)

    # Close the box by repeating the first vertex at the end
    verts = torch.cat([verts, verts[:1]], dim=0)
    ax.plot(verts[:, 0], verts[:, 1], linestyle='-', marker='', c=c, linewidth=linewidth, **kwargs)
    return ax

def sort_vertices_counterclockwise(vertices: torch.Tensor) -> torch.Tensor:
    centroid = vertices.mean(dim=0)
    angles = torch.atan2(vertices[:,1] - centroid[1], vertices[:,0] - centroid[0])
    sorted_idx = torch.argsort(angles)
    return vertices[sorted_idx]

def plot_2d_partition(ax, partition: dd_schemes.GridPartition, c: str = 'blue', linewidth: float = 2, **kwargs):
    for i in range(partition.shape[0]):
        for j in range(partition.shape[1]):
            cell = partition[i, j]
            if cell is not None:
                try: 
                    domain = cell.domain
                except:
                    domain = cell.domain

                ax = plot_2d_cell(ax, cell.domain, c=c, linewidth=linewidth, **kwargs)
    return ax


def plot_2d_basis(ax, offset: torch.Tensor, mat: torch.Tensor, color: str = 'blue', linewidth: float = 3.):
    style = ['solid', 'dashed']
    for i in range(2):
        ax.arrow(
            *offset, mat[0, i], mat[1, i],
            head_width=0.1, head_length=0.1, fc=color, ec=color,
            length_includes_head=True,
            linewidth=linewidth, linestyle=style[i]
        )
    ax.set_aspect('equal')
    return ax


def set_axis(ax, xlim=None, ylim=None):
    xlims = ax.get_xlim() if xlim is None else xlim
    ylims = ax.get_ylim() if ylim is None else ylim
    min_lim = min(xlims[0], ylims[0])
    max_lim = max(xlims[1], ylims[1])
    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)
    return ax