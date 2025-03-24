import torch
from matplotlib import pyplot as plt
import bound_propagation as bp
import linear_bound_propagation as lbp
import math


def plot_batch(ax, x, y, *args, **kwargs):
    for idx in range(x.size(0)):
        ax.plot(x[idx], y[idx], *args, **kwargs)


@torch.no_grad()
def test_strict_crown_ibp_1d(f, locs, l, u, l_plot, u_plot):
    lb_neg = f.strict_crown_ibp(bp.HyperRectangle(torch.full_like(locs, l), locs), locs)
    lb_pos = f.strict_crown_ibp(bp.HyperRectangle(locs, torch.full_like(locs, u)), locs)

    x = torch.linspace(l_plot, u_plot, 100).unsqueeze(-1)
    x_neg = torch.cat([torch.linspace(max(l, l_plot), loc.squeeze(), 100).view(1, -1, 1) for loc in locs], dim=0)
    x_pos = torch.cat([torch.linspace(loc.squeeze(), min(u_plot, u), 100).view(1, -1, 1) for loc in locs], dim=0)

    y_locs = f(locs)
    y = f(x)

    y_neg_lower = torch.einsum('...ij,bnj->bni', lb_neg.lower[0], x_neg) + lb_neg.lower[1]
    y_neg_upper = torch.einsum('...ij,bnj->bni', lb_neg.upper[0], x_neg) + lb_neg.upper[1]

    y_pos_lower = torch.einsum('...ij,bnj->bni', lb_pos.lower[0], x_pos) + lb_pos.lower[1]
    y_pos_upper = torch.einsum('...ij,bnj->bni', lb_pos.upper[0], x_pos) + lb_pos.upper[1]

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


@torch.no_grad()
def test_strict_crown_ibp_nd(f, locs, l, u):
    lb_neg = f.strict_crown_ibp(bp.HyperRectangle(torch.full_like(locs, l), locs), locs)
    lb_pos = f.strict_crown_ibp(bp.HyperRectangle(locs, torch.full_like(locs, u)), locs)

if __name__ == "__main__":
    ## 1D Illustration Clamp
    class SigmoidDynamics(torch.nn.Sequential):
        def __init__(self):
            super().__init__(lbp.Linear(torch.tensor([[1.0]])), torch.nn.Sigmoid())

    loc = 2.0
    inf = 8.
    region_width = 6.

    test_strict_crown_ibp_1d(
        f=lbp.factory.build(SigmoidDynamics()),
        locs=torch.tensor([loc]).unsqueeze(0),
        l=loc - inf,
        u=loc + inf,
        l_plot=loc - region_width,
        u_plot=loc + region_width)

    ## 1D Illustration Sinusoid
    class SinDynamics(torch.nn.Sequential):
        def __init__(self):
            super().__init__(lbp.Linear(torch.tensor([[1.0]])), bp.Sin())

    loc = 0.2*math.pi
    inf = 1.1 * math.pi
    region_width = 1. * math.pi

    test_strict_crown_ibp_1d(
        f=lbp.factory.build(SinDynamics()),
        locs=torch.tensor([loc]).unsqueeze(0),
        l=loc - inf,
        u=loc + inf,
        l_plot=loc - region_width,
        u_plot=loc + region_width)

    ## 1D Illustration Saturation
    class SaturationDynamics(torch.nn.Sequential):
        def __init__(self):
            super().__init__(lbp.Linear(torch.tensor([[0.9]])), bp.Clamp(min=-0.5, max=0.5))

    loc = 1.0
    inf = 5.
    region_width = 3.

    test_strict_crown_ibp_1d(
        f=lbp.factory.build(SaturationDynamics()),
        locs=torch.tensor([loc]).unsqueeze(0),
        l=loc - inf,
        u=loc + inf,
        l_plot=loc - region_width,
        u_plot=loc + region_width)

    ## 2D LinearSigmoid dynamics with full matrix, but we can still handle!
    class TractableLinearSigmoidDynamics(torch.nn.Sequential):
        def __init__(self):
            super().__init__(lbp.Linear(torch.tensor([[1., 3.],[2., 1.]])), torch.nn.Sigmoid(), bp.Clamp(min=-0.5, max=0.5))

    loc = 2.0
    inf = 10.

    test_strict_crown_ibp_nd(
        f=lbp.factory.build(TractableLinearSigmoidDynamics()),
        locs=torch.tensor([loc, loc]).unsqueeze(0),
        l=loc - inf,
        u=loc + inf)


    ## 2D LinearSigmoid dynamics with full matrix, which we can not handle
    class IntractableLinearSigmoidDynamics(torch.nn.Sequential):
        def __init__(self):
            super().__init__(lbp.Linear(torch.tensor([[1., -3.],[2., 1.]])), bp.Clamp(min=-0.5, max=0.5))

    try:
        test_strict_crown_ibp_nd(
            f=lbp.factory.build(IntractableLinearSigmoidDynamics()),
            locs=torch.tensor([loc, loc]).unsqueeze(0),
            l=loc - inf,
            u=loc + inf)
    except Exception as e:
        print(f"The algorithm failed for this case, as expected. Error message: {e}")


