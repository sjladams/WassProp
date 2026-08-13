from typing import Union

import torch
import bound_propagation as bp
import pointwise_lipschitz as pl

from .dynamics import Dynamics, StochasticDynamics

bp_factory = bp.factory.BoundModelFactory()
pl_factory = pl.factory.BoundModelFactory()

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
    try: # TODO temp fix
        ibp_bounds_f = factory.build(f).ibp(bp.HyperRectangle(l, u))
        f_c = f(locs)
        beta = torch.max(
            torch.linalg.vector_norm(ibp_bounds_f.lower - f_c, dim=-1, ord=2).pow(2),
            torch.linalg.vector_norm(ibp_bounds_f.upper - f_c, dim=-1, ord=2).pow(2)
        )
    except Exception as e:
        print(f"Warning: could not compute global_ibp_sq_norm_fx_fc due to {e}. Returning Lipschitz bound.")
        beta = torch.full(locs.shape[:-1], torch.inf)

    return beta


def global_lbp_sq_norm_fx_fc(
        f: Union[StochasticDynamics, Dynamics],
        locs: torch.Tensor) -> torch.Tensor:
    """
    find vector a such that ||f(x) - f(c_i)||^2 leq a_i||x-c_i||^2 for all x and c_i the loc of region R_i

    :param f: dynamics
    :param locs: batch of c's with shape (num_locs, num_dims)
    """

    global_lipschitz = torch.as_tensor(f.global_lipschitz)
    pointwise_lipschitz = pl_factory.build(f).pointwise_lipschitz(locs)

    return torch.min(global_lipschitz.expand(locs.shape[:-1]), pointwise_lipschitz).pow(2)
