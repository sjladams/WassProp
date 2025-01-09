import matplotlib.pyplot as plt
import numpy as np
import torch

plt.style.use('seaborn-v0_8-bright')

plt.rcParams.update({
    'font.size': 30,
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
def plot_multi_step(dynamics, samples: dict):

    colors = plt.cm.tab10(np.linspace(0, 1, len(samples.keys())))

    if dynamics.num_state_dims == 1:
        p_samples = torch.stack([samples[k]['p'] for k in samples.keys()])
        q_samples = torch.stack([samples[k]['q'] for k in samples.keys()])

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 12))
        for k in list(samples.keys()):

            ax[0].hist(p_samples[k, :, 0], color=colors[k], bins=20, density=True, label=rf'$t={k+1}$')
            ax[1].hist(q_samples[k, :, 0], color=colors[k], bins=20, density=True, label=rf'$t={k+1}$')

        ax[0].set_title(r'$\mathbb{P}_{x_t}$ (actual distr.)')
        ax[1].set_title(r'$\hat{\mathbb{P}}_{x_t}$ (our approx.)')

        ax[0].set_xlabel(r'$x_t$')
        ax[1].set_xlabel(r'$x_t$')
        ax[0].set_ylabel('Frequency')

        ax[0].grid(True)
        ax[1].grid(True)

        ax[0].set_xlim(0, 2)
        ax[1].set_xlim(0, 2)
        ax[0].set_xlim(min(ax[0].get_xlim()[0], ax[1].get_xlim()[0]),
                         max(ax[0].get_xlim()[1], ax[1].get_xlim()[1]))
        ax[1].set_xlim(ax[0].get_xlim()[0], ax[0].get_xlim()[1])
        ax[0].set_ylim(min(ax[0].get_ylim()[0], ax[1].get_ylim()[0]),
                         max(ax[0].get_ylim()[1], ax[1].get_ylim()[1]))
        ax[1].set_ylim(ax[0].get_ylim()[0], ax[0].get_ylim()[1])
        #ax[0].axis('equal')
        #ax[1].axis('equal')
        ax[0].legend(loc='upper left')
        ax[1].legend(loc='upper left')

        plt.savefig(r'C:\Users\efigueiredomot\Desktop\Papers\Wasserstein\comparison_approx_sigmoid.pdf', format='pdf')
        plt.show()

    elif dynamics.num_state_dims == 2:
        p_samples = torch.stack([samples[k]['p'] for k in samples.keys()])
        q_samples = torch.stack([samples[k]['q'] for k in samples.keys()])

        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(14, 14))
        for k in list(samples.keys()):
            # Plot using hist2d with color intensity indicating the density
            ax[0][0].hist2d(p_samples[k,:,0], p_samples[k,:,1], bins=100, cmap=COLORS[k], alpha=0.8, cmin=0.1, label=rf'$t={k+1}$')
            ax[0][1].hist2d(q_samples[k,:,0], q_samples[k,:,1], bins=100, cmap=COLORS[k], alpha=0.8, cmin=0.1, label=rf'$t={k+1}$')

            # Plot only first dimension
            ax[1][0].hist(p_samples[k,:,0], color=colors[k], bins=100, density=True, label=rf'$t={k+1}$')
            ax[1][1].hist(q_samples[k, :, 0], color=colors[k], bins=100, density=True, label=rf'$t={k+1}$')

            # Plot only second dimension
            ax[2][0].hist(p_samples[k, :, 1], color=colors[k], bins=100, density=True, label=rf'$t={k+1}$')
            ax[2][1].hist(q_samples[k, :, 1], color=colors[k], bins=100, density=True, label=rf'$t={k+1}$')

            # cb = plt.colorbar(label='Point Density')
            # plt.legend(loc='lower right')

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

        ax[0][0].set_xlim(min(ax[0][0].get_xlim()[0], ax[0][1].get_xlim()[0]),
                       max(ax[0][0].get_xlim()[1], ax[0][1].get_xlim()[1]))
        ax[0][1].set_xlim(ax[0][0].get_xlim()[0], ax[0][0].get_xlim()[1])
        ax[0][0].set_ylim(min(ax[0][0].get_ylim()[0], ax[0][1].get_ylim()[0]),
                       max(ax[0][0].get_ylim()[1], ax[0][1].get_ylim()[1]))
        ax[0][1].set_ylim(ax[0][0].get_ylim()[0], ax[0][0].get_ylim()[1])
        ax[0][0].axis('equal')
        ax[0][1].axis('equal')
        plt.show()
    else:
        raise NotImplementedError

@torch.no_grad()
def plot_dict(dict,
              axis_x,
              label_names,
              x_label : str = "Number of locations",
              y_label : str = r"$\mathbb{W}_{\rho}(f \# \mathbb{P}, f \# \Delta_{\mathcal{R}, \mathcal{C}} \# \mathbb{P})$",
              log_scale : bool = True):

    colors = plt.cm.tab10(np.linspace(0, 1, len(dict)))

    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    for i, (key, values) in enumerate(dict.items()):
        plt.scatter(axis_x, values, color=colors[i], label=label_names[i])
        plt.plot(axis_x, values, color=colors[i], linestyle='--', linewidth=1)

    if log_scale:
        plt.xscale('log')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend(
        loc='center left', bbox_to_anchor=(1, 0.5), frameon=False
    )
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    #plt.savefig('comparison_approx_sigmoid.pdf', format='pdf')
    plt.show()
