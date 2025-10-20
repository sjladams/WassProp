import matplotlib.pyplot as plt
import torch
import pprint
import discretize_distributions as dd
import numpy as np
from scipy.stats import norm

from duq_via_wasserstein import single_step, multi_step, multi_step_empirical, SampledPath, AmbiguityBall
import duq_via_wasserstein.wasserstein as wasserstein

from dynamics import get_stoch_dynamics
from handlers import parse_arguments
import utils


COLORS = ['Blues', 'BuPu', 'PuRd', 'Greens', 'Oranges', 'Reds', 'Greys', 'Purples', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
COLORS_HIST = ['#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30']


def sigmoid_example(): # TODO test
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
    plt.show()

def convergence_analysis(dynamics_type, setting, num_locs, w2_p__q):
    args = parse_arguments(
        dynamics_type=dynamics_type,
        dynamics_setting=setting,
        num_locs=num_locs,
    )

    dynamics = get_stoch_dynamics(name=args.dynamics_type, **vars(args.dynamics))
    initial_dist = utils.get_initial_dist(loc=args.initial_dist.loc, variance=args.initial_dist.variance)
    noise_dist = utils.get_noise_dist(loc=args.noise_dist.loc, variance=args.noise_dist.variance)

    results = dict()
    for method in ['global_lipschitz', 'lagrangian_duality']:
        w2 = single_step(
            dynamics=dynamics,
            q=AmbiguityBall(initial_dist, w2_p__q),
            noise=AmbiguityBall(noise_dist, 0.),
            num_locs=args.num_locs,
            use_lagrangian_duality=True
        ).w2
        results[method] = float(w2)
    return results


def num_loc_convergence_analysis(w2_p__q: float):
    num_locs_experiment = [10, 100, 1000, 10000]

    run_inputs = { # [dynamics_type, dynamics_setting]
        'Sigmoid (1D)' : ('SigmoidDynamics', 0),
        'Bounded Linear (2D)' : ('BoundedLinearDynamics', 0),
        'Quadruple-Tank (4D)' : ('LinearDynamics', 0),
        'NN Layer (10D)' : ('DiagonalSigmoidDynamics', 2),
        # 'Mountain Car (2D)' : ('MountainCarDynamics', 0),
        # 'Dubins car (3D)' : ('DubinsCarDynamics', 0)
    }

    results = {}
    for dynamics_name, (dynamics_type, setting) in run_inputs.items():
        results[dynamics_name] = {}
        for num_locs in num_locs_experiment:
            results[dynamics_name][num_locs] = convergence_analysis(dynamics_type, setting, num_locs, w2_p__q)

    pprint.pprint(results)

def w2_p__q_convergence_analysis():
    w2_p__q_options = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    run_inputs = { # (dynamics_type, dynamics_setting, num_locs)
        'Sigmoid (1D)' : ('SigmoidDynamics', 0, 100),
        'Bounded Linear (2D)' : ('BoundedLinearDynamics', 0, 100),
        'Quadruple-Tank (4D)' : ('LinearDynamics', 0, 1000),
        'NN Layer (10D)' : ('DiagonalSigmoidDynamics', 2, 1000),
        # 'Mountain Car (2D)' : ('MountainCarDynamics', 0, 100),
        # 'Dubins car (3D)' : ('DubinsCarDynamics', 0, 1000)
    }

    results = {}
    for dynamics_name, (dynamics_type, setting, num_locs) in run_inputs.items():
        results[dynamics_name] = {}
        for w2_p__q in w2_p__q_options:
            results[dynamics_name][w2_p__q] = convergence_analysis(dynamics_type, setting, num_locs, w2_p__q)

    pprint.pprint(results)


def mountain_car_mc_plot(): # TODO fix color scheme
    args = parse_arguments(
        dynamics_type='MountainCarDynamics',
        dynamics_setting=0,
        num_locs=100,
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
    ).detach()
    approx_samples = SampledPath({k: path.at(k).sample(args.num_samples) for k in path.ordered_indices}).detach()


    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(24, 36))
    for k in true_samples.ordered_indices:
        # Plot using hist2d with color intensity indicating the density
        ax[0,0].hist2d(true_samples.at(k)[:,0], true_samples.at(k)[:,1], bins=100, cmap=COLORS[k], alpha=0.8, cmin=0.1, label=rf'$t={k+1}$')
        ax[0,1].hist2d(approx_samples.at(k)[:,0], approx_samples.at(k)[:,1], bins=100, cmap=COLORS[k], alpha=0.8, cmin=0.1, label=rf'$t={k+1}$')

        # Plot only first dimension
        ax[1,0].hist(true_samples.at(k)[:,0], color=COLORS_HIST[k], bins=50, density=True, label=rf'$t={k+1}$')
        ax[1,1].hist(approx_samples.at(k)[:,0], color=COLORS_HIST[k], bins=50, density=True, label=rf'$t={k+1}$')

        # Plot only second dimension
        ax[2,0].hist(true_samples.at(k)[:, 1], color=COLORS_HIST[k], bins=50, density=True, label=rf'$t={k+1}$')
        ax[2,1].hist(approx_samples.at(k)[:, 1], color=COLORS_HIST[k], bins=50, density=True, label=rf'$t={k+1}$')

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


if __name__ == '__main__':
    torch.manual_seed(0)

    # # Figure 3
    # sigmoid_example()

    # Figure 5
    num_loc_convergence_analysis(w2_p__q=0.0)
    num_loc_convergence_analysis(w2_p__q=0.1)

    # Figure 6
    w2_p__q_convergence_analysis()

    # Figure 7
    mountain_car_mc_plot()