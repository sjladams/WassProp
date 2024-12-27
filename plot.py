import matplotlib.pyplot as plt
import torch

from bound import global_lbp_sq_norm_fx_fc, global_ibp_sq_norm_fx_fc

COLORS = ['Blues', 'BuPu', 'PuRd', 'Greens', 'Oranges', 'Reds', 'Greys', 'Purples',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

def plot_dynamics(dynamics):

    if dynamics.num_dims == 1:
        # Plot dynamics
        x = torch.linspace(start=-5, end=5, steps=500).view(-1, 1)
        y = dynamics(x)

        fig_dynamics = plt.figure()
        plt.plot(x, y)
        plt.title("Dynamics f(x)")
        plt.show()

@torch.no_grad()
def plot_single_step(dynamics, w2_bounds: dict, **kwargs):
    w2_p__q_options = list(w2_bounds.keys())

    fig_w2_bounds = plt.figure()
    for key in w2_bounds[w2_p__q_options[0]].keys():
        if not key in ['sign_q']:
            plt.plot(w2_p__q_options, [w2_bounds[w2_p__q][key] for w2_p__q in w2_p__q_options], label=key)

    plt.legend()
    plt.title(f"{dynamics.__class__.__name__} (Lipschitz={dynamics.global_lipschitz:.2f})")
    plt.xlabel('$W_2(p,q)$')
    plt.xticks(w2_p__q_options)
    plt.ylabel(r'$W_2(f p, f \Delta q)$')
    plt.xlim(min(w2_p__q_options), max(w2_p__q_options))
    plt.show()


@torch.no_grad()
def plot_multi_step(dynamics, samples: dict):
    if dynamics.num_dims == 1:
        raise NotImplementedError
    elif dynamics.num_dims == 2:
        p_samples = torch.stack([samples[k]['p'] for k in samples.keys()])
        q_samples = torch.stack([samples[k]['q'] for k in samples.keys()])

        fig, ax = plt.subplots(nrows=1, ncols=2)
        for k in list(samples.keys()):
            # Plot using hist2d with color intensity indicating the density
            ax[0].hist2d(p_samples[k,:,0], p_samples[k,:,1], bins=100, cmap=COLORS[k], alpha=0.8, cmin=0.1)
            ax[1].hist2d(q_samples[k,:,0], q_samples[k,:,1], bins=100, cmap=COLORS[k], alpha=0.8, cmin=0.1)

            # cb = plt.colorbar(label='Point Density')
            # plt.legend(loc='lower right')

        ax[0].set_title('p')
        ax[1].set_title('q')
        ax[0].set_xlabel('State[0]')
        ax[1].set_xlabel('State[0]')
        ax[0].set_ylabel('State[1]')
        ax[0].grid(True)
        ax[1].grid(True)
        ax[0].set_xlim(min(ax[0].get_xlim()[0], ax[1].get_xlim()[0]),
                           max(ax[0].get_xlim()[1], ax[1].get_xlim()[1]))
        ax[1].set_xlim(ax[0].get_xlim()[0], ax[0].get_xlim()[1])
        ax[0].set_ylim(min(ax[0].get_ylim()[0], ax[1].get_ylim()[0]),
                           max(ax[0].get_ylim()[1], ax[1].get_ylim()[1]))
        ax[1].set_ylim(ax[0].get_ylim()[0], ax[0].get_ylim()[1])
        ax[0].axis('equal')
        ax[1].axis('equal')
        plt.show()
    else:
        raise NotImplementedError

@torch.no_grad()
def plot_norm_overapproximation(dynamics, signature, loc_pos, **kwargs):

    alpha = global_lbp_sq_norm_fx_fc(dynamics, signature.locs)
    beta = global_ibp_sq_norm_fx_fc(dynamics, signature.locs).upper.squeeze(-1)

    start, end, steps = signature.locs[loc_pos]-2, signature.locs[loc_pos]+2, 1000
    grid_x, grid_y = torch.meshgrid(torch.linspace(start[0], end[0], steps), torch.linspace(start[1], end[1], steps), indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1)

    f_grid = dynamics(grid)
    f_locs = dynamics(signature.locs)

    f_grid_sub_f_c = f_grid - f_locs[loc_pos]

    f_squared_norms = torch.sum(f_grid_sub_f_c ** 2, dim=-1).reshape(-1).numpy()
    x_squared_norms = torch.sum((grid - signature.locs[loc_pos]) ** 2, dim=-1).reshape(-1).numpy()

    norm_overapprox_alpha = alpha[loc_pos] * x_squared_norms
    norm_overapprox_beta = 0 * x_squared_norms + beta[loc_pos].item()

    plt.figure(figsize=(8, 6))
    plt.scatter(x_squared_norms, f_squared_norms, color='blue', s=1)
    plt.scatter(x_squared_norms, norm_overapprox_alpha, color='red', s=1)
    plt.scatter(x_squared_norms, norm_overapprox_beta, color='orange', s=1)

    plt.show()