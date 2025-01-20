import matplotlib.pyplot as plt
import torch


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
    if dynamics.num_state_dims == 1:
        raise NotImplementedError
    elif dynamics.num_state_dims == 2:
        p_samples = torch.stack([samples[k]['p'] for k in samples.keys()])
        q_samples = torch.stack([samples[k]['q'] for k in samples.keys()])

        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(14, 14))
        for k in list(samples.keys()):
            # Plot using hist2d with color intensity indicating the density
            ax[0][0].hist2d(p_samples[k,:,0], p_samples[k,:,1], bins=100, cmap=COLORS[k], alpha=0.8, cmin=0.1)
            ax[0][1].hist2d(q_samples[k,:,0], q_samples[k,:,1], bins=100, cmap=COLORS[k], alpha=0.8, cmin=0.1)

            # Plot only first dimension
            ax[1][0].hist(p_samples[k,:,0], color=COLORS_HIST[k], bins=100, density=True)
            ax[1][1].hist(q_samples[k, :, 0], color=COLORS_HIST[k], bins=100, density=True)

            # Plot only second dimension
            ax[2][0].hist(p_samples[k, :, 1], color=COLORS_HIST[k], bins=100, density=True)
            ax[2][1].hist(q_samples[k, :, 1], color=COLORS_HIST[k], bins=100, density=True)

            # cb = plt.colorbar(label='Point Density')
            # plt.legend(loc='lower right')

        ax[0][0].set_title('p')
        ax[0][1].set_title('q')

        ax[0][0].set_xlabel('State[0]')
        ax[1][0].set_xlabel('State[0]')
        ax[0][0].set_ylabel('State[1]')

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
