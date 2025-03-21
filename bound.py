from typing import Union
import torch
import bound_propagation as bp

from dynamics import Dynamics, StochasticDynamics, Separable, CompositionalStructure
from linear_bound_propagation import factory as linear_factory

factory = bp.BoundModelFactory()


def global_ibp_sq_norm_fx_fc(f: torch.nn.Sequential, locs: torch.Tensor) -> torch.Tensor:
    """
    find vector b such that ||f(x) - f(c_i)||^2 leq beta_i for all x  and c_i the loc of region R_i

    :param f: dynamics
    :param locs: c_i's
    """
    inf = 1e6 # bound_propagation does not support inf, instead use a large value

    l = torch.full_like(locs, -inf)
    u = torch.full_like(locs, inf)

    # Alternative (cleaner) implementation:
    ibp_bounds_f = factory.build(f).ibp(bp.HyperRectangle(l, u))
    f_c = f(locs)
    beta = torch.max(
        torch.linalg.vector_norm(ibp_bounds_f.lower - f_c, dim=-1, ord=2).pow(2),
        torch.linalg.vector_norm(ibp_bounds_f.upper - f_c, dim=-1, ord=2).pow(2)
    )

    return beta


def _global_lbp_sq_norm_fx_fc_quadrant(
        f: Union[StochasticDynamics, Dynamics],
        locs: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor) -> torch.Tensor:

    input_bound = bp.HyperRectangle(lower, upper)
    lb = linear_factory.build(f).strict_crown_ibp(input_bound, locs)

    # From linear bounds to bounds on the norms:
    alpha = torch.max(
        torch.svd(lb.lower[0]).S.max(-1).values,
        torch.svd(lb.upper[0]).S.max(-1).values
    ).pow(2)

    return alpha


def global_lbp_sq_norm_fx_fc(
        f: Union[StochasticDynamics, Dynamics],
        locs: torch.Tensor) -> torch.Tensor:
    """
    find vector a such that ||f(x) - f(c_i)||^2 leq a_i||x-c_i||^2 for all x and c_i the loc of region R_i

    :param f: dynamics
    :param locs: batch of c's with shape (num_locs, num_dims)
    """

    if isinstance(f, CompositionalStructure):
        alphas = []
        for component in f.children():
            alphas.append(global_lbp_sq_norm_fx_fc_component(component, locs))
            locs = component(locs)
        return torch.stack(alphas, dim=-1).prod(dim=-1)
    else:
        return global_lbp_sq_norm_fx_fc_component(f, locs)


def global_lbp_sq_norm_fx_fc_component(
        f: Union[StochasticDynamics, Dynamics],
        locs: torch.Tensor) -> torch.Tensor:
    """
    :param f: dynamics
    :param locs: batch of c's with shape (num_locs, num_dims)
    """

    quadrants = generate_quadrants(
        num_dims=f.num_dims,
        only_pos_and_neg=isinstance(f, Separable)
    )

    inf = 1e2 # bound_propagation does not support inf, instead use a large value
    quadrants = quadrants.clip(-inf, inf)

    num_locs = locs.shape[-2]
    quadrants = quadrants.unsqueeze(-2).repeat(1, 1, num_locs, 1) + locs

    alphas = torch.zeros(len(quadrants), num_locs).fill_(torch.nan)
    for idx, quadrant in enumerate(quadrants):
        alphas[idx] = _global_lbp_sq_norm_fx_fc_quadrant(f, locs, quadrant[0], quadrant[1])

    alpha = alphas.max(dim=0).values.clamp(min=0., max=f.global_lipschitz**2)

    return alpha



def generate_quadrants(num_dims, only_pos_and_neg: bool = False) -> torch.Tensor:
    """
    :param only_pos_and_neg:
    :return: quadrants of shape (nr_quadrants, 2, num_dims)
    """
    if only_pos_and_neg:
        # If the dynamics is separable, then we only have to the positive and negative quadrants
        quadrants = torch.stack((
            torch.stack((torch.full( (num_dims,), -torch.inf), torch.zeros(num_dims))),
            torch.stack((torch.zeros(num_dims), torch.full((num_dims,), torch.inf)))
        ))
    else:
        # else, we have to consider all quadrants:
        # Generate all combinations of signs (+1 and -1) for each dimension
        signs = torch.cartesian_prod(*[torch.tensor([-1, 1]) for _ in range(num_dims)])

        # Compute lower and upper bounds for each quadrant
        lower = torch.where(
            signs == torch.tensor(-1),
            torch.full_like(signs, -torch.inf, dtype=torch.float32),
            torch.zeros_like(signs)
        )
        upper = torch.where(
            signs == torch.tensor(1),
            torch.full_like(signs, torch.inf, dtype=torch.float32),
            torch.zeros_like(signs)
        )

        # Stack lower and upper bounds into shape (nr_quadrants, 2, n)
        quadrants = torch.stack([lower, upper], dim=1)

    return quadrants