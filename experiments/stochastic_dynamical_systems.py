import matplotlib.pyplot as plt
import numpy as np
import torch

import torch

from duq_via_wasserstein import multi_step, multi_step_empirical, SampledPath, AmbiguityBall

from dynamics import get_stoch_dynamics
from handlers import parse_arguments
import utils


@torch.no_grad()
def plot_multi_step(dynamics, true_samples: SampledPath, approx_samples: SampledPath, layout: str = "hist"):
    if not dynamics.num_state_dims in [1, 2]:
        raise NotImplementedError("Only implemented for 1D and 2D dynamics")

    colors = plt.cm.tab10(np.linspace(0, 1, len(true_samples.ordered_indices)))

    if layout == "scatter":
        fig, ax = plt.subplots(figsize=(12, 12))
        for k in true_samples.ordered_indices:
            ax.scatter(true_samples.at(k)[:,0], true_samples.at(k)[:,1], label=rf'$t={k+1}$')
        ax.legend(loc='upper left')
        ax.set_xlim([-1., 1.])
        ax.set_ylim([-1., 1.])
    elif layout == "hist":
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(24, 36))
        for k in true_samples.ordered_indices:

            # Plot using hist2d with color intensity indicating the density
            ax[0][0].hist2d(true_samples.at(k)[:,0], true_samples.at(k)[:,1], bins=100, cmap=colors[k], alpha=0.8, cmin=0.1, label=rf'$t={k+1}$')
            ax[0][1].hist2d(approx_samples.at(k)[:,0], approx_samples.at(k)[:,1], bins=100, cmap=colors[k], alpha=0.8, cmin=0.1, label=rf'$t={k+1}$')

            # Plot only first dimension
            ax[1][0].hist(true_samples.at(k)[:,0], color=colors[k], bins=50, density=True, label=rf'$t={k+1}$')
            ax[1][1].hist(approx_samples.at(k)[:,0], color=colors[k], bins=50, density=True, label=rf'$t={k+1}$')

            # Plot only second dimension
            ax[2][0].hist(true_samples.at(k)[:, 1], color=colors[k], bins=50, density=True, label=rf'$t={k+1}$')
            ax[2][1].hist(approx_samples.at(k)[:, 1], color=colors[k], bins=50, density=True, label=rf'$t={k+1}$')

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


def multistep_approximation(dynamics_type, setting, num_locs):
    args = parse_arguments(
        dynamics_type=dynamics_type,
        dynamics_setting=setting,
        num_locs=num_locs,
        num_samples=10000
    )

    num_time_steps = 2

    dynamics = get_stoch_dynamics(name=args.dynamics_type, **vars(args.dynamics))
    initial_dist = utils.get_initial_dist(loc=args.initial_dist.loc, variance=args.initial_dist.variance)
    noise_dist = utils.get_noise_dist(loc=args.noise_dist.loc, variance=args.noise_dist.variance)

    q = AmbiguityBall(initial_dist, 0.1)
    noise = AmbiguityBall(noise_dist, 0.0)

    path = multi_step(
        dynamics=dynamics, 
        q=q, 
        noise=noise,
        num_time_steps=num_time_steps,
        use_lagrangian_duality=True,
        num_locs=args.num_locs,
    )

    true_samples = multi_step_empirical(
        dynamics=dynamics,
        p_emp=q.sample(args.num_samples),
        noise=noise,
        num_time_steps=num_time_steps,
    )
    approx_samples = SampledPath({k: path.at(k).sample(args.num_samples) for k in path.ordered_indices})

    plot_multi_step(dynamics=dynamics, true_samples=true_samples, approx_samples=approx_samples)


if __name__ == '__main__':
    torch.manual_seed(0)

    dynamics_type = 'MountainCarDynamics'
    num_locs = 100
    dyn_setting = 0

    multistep_approximation(dynamics_type, dyn_setting, num_locs)