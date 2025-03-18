import torch
from matplotlib import pyplot as plt
import bound_propagation as bp


from bound import global_lbp_sq_norm_fx_fc, global_ibp_sq_norm_fx_fc
import dynamics
import discretize_distributions as ds
from regions import HyperRectangularVoronoiPartition

from experiments import multi_step, get_noise_dist, get_initial_dist, single_step_w2_options
from dynamics import get_dynamics

from utils import load_params, parse_arguments

from linear_bound_propagation import Linear, factory as linear_factory

factory = bp.BoundModelFactory()


def plot_batch(ax, x, y, *args, **kwargs):
    for idx in range(x.size(0)):
        ax.plot(x[idx], y[idx], *args, **kwargs)


@torch.no_grad
def find_alpha(f, locs, l, u, l_plot, u_plot):
    lb_neg = f.crown_ibp(bp.HyperRectangle(torch.ones_like(locs).fill_(l), locs))
    lb_pos = f.crown_ibp(bp.HyperRectangle(locs, torch.ones_like(locs).fill_(u)))

    x = torch.linspace(l_plot, u_plot, 100).unsqueeze(-1)
    x_neg = torch.cat([torch.linspace(l_plot, loc.squeeze(), 100).view(1, -1, 1) for loc in locs], dim=0)
    x_pos = torch.cat([torch.linspace(loc.squeeze(), u_plot, 100).view(1, -1, 1) for loc in locs], dim=0)

    y_locs = f(locs)

    y = f(x)

    y_neg_lower = torch.einsum('...ij,bnj->bni', lb_neg.lower[0], x_neg) + lb_neg.lower[1]
    y_neg_upper = torch.einsum('...ij,bnj->bni', lb_neg.upper[0], x_neg) + lb_neg.upper[1]

    y_pos_lower = torch.einsum('...ij,bnj->bni', lb_pos.lower[0], x_pos) + lb_pos.lower[1]
    y_pos_upper = torch.einsum('...ij,bnj->bni', lb_pos.upper[0], x_pos) + lb_pos.upper[1]

    # assert ((y_neg_lower[:, -1] - y_locs).abs() < 1e-6).all()
    # assert ((y_neg_upper[:, -1] - y_locs).abs() < 1e-6).all()
    # assert ((y_pos_lower[:, 0] - y_locs).abs() < 1e-6).all()
    # assert ((y_pos_upper[:, 0] - y_locs).abs() < 1e-6).all()

    fig, ax = plt.subplots()

    ax.plot(x, y, 'b')
    plot_batch(ax, locs, y_locs, 'k*')
    plot_batch(ax, x_neg, y_neg_upper, 'g')
    plot_batch(ax, x_pos, y_pos_upper, 'g')
    plot_batch(ax, x_neg, y_neg_lower, 'r')
    plot_batch(ax, x_pos, y_pos_lower, 'r')
    ax.set_xlim(x[0], x[-1])

    ax.grid(True)
    plt.show()


def investigate_linear_bound_propagation():
    ## Linear Sigmoid Dynamics:
    class LinearDiagonalSigmoidDynamics(torch.nn.Sequential):
        def __init__(self):
            super(LinearDiagonalSigmoidDynamics, self).__init__(Linear(torch.eye(1)*5), torch.nn.Sigmoid())

    ## Bounded Linear Dynamics:
    class LinearBoundedDynamics(torch.nn.Sequential):
        def __init__(self):
            super(LinearBoundedDynamics, self).__init__(Linear(torch.eye(1)*2), bp.Clamp(-2, 2))

    class BoundedLinearDynamics(torch.nn.Sequential):
        def __init__(self):
            super(BoundedLinearDynamics, self).__init__(bp.Clamp(-2, 2), Linear(torch.eye(1)*2))

    ## Bounded Linear Dynamics:
    class SinLinearDynamics(torch.nn.Sequential):
        def __init__(self):
            super(SinLinearDynamics, self).__init__(Linear(torch.eye(1) * 0.5), bp.Sin())


    ## Test Linear Bounding:
    # f = linear_factory.build(LinearDiagonalSigmoidDynamics())
    # f = linear_factory.build(LinearBoundedDynamics())
    # f = linear_factory.build(SinLinearDynamics())
    f = linear_factory.build(BoundedLinearDynamics())

    # shift = 2*math.pi
    shift = 0.
    l, u = -torch.inf, torch.inf
    # l, u = -10, 10
    l_plot, u_plot = -4., 4.
    # l_plot, u_plot = -2.5, 2.5
    # l_plot, u_plot = -2.5*math.pi, 4.5*math.pi

    loc = -3.

    # loc = 1.99/4 * math.pi
    # loc = 3/4 * math.pi
    # loc = 5/4 * math.pi
    # loc = 7/4 * math.pi
    locs = torch.linspace(loc, loc, 1).unsqueeze(-1)

    # find_alpha(f, locs + shift, l + shift, u + shift, l_plot + shift, u_plot + shift)
    find_alpha(f, locs, l, u, l_plot, u_plot)


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


def invest_alphas_betas():
    torch.manual_seed(0)

    args = parse_arguments(
        dynamics_type="MountainCarDynamics",
        num_dims=2,
        dynamics_setting=0,
        num_locs=100,
        num_locs_after_compr=1,
        num_samples=5000,
        lr=0.01,
        num_iterations=1000,
        plot=False
    )

    params = load_params(args)

    dynamics = get_dynamics(**params)
    dist = get_initial_dist(**params)

    signature = ds.discretization_generator(dist=dist, num_locs=args.num_locs)

    plot_norm_overapproximation(dynamics, signature, 76)


if __name__ == "__main__":
    invest_alphas_betas()


