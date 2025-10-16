from typing import Callable, Dict, Protocol, Optional, Union

import torch
import bound_propagation as bp
import math

from sympy.solvers.solveset import NonlinearError

from . import linear_bound_propagation as lbp


class StochasticDynamics(torch.nn.Sequential):
    """
    x_{k+1} = stochastic_dynamics(x_k, noise_k)
    """
    def __init__(self, num_state_dims: int, num_noise_dims: int, modules: list):
        self.num_state_dims = num_state_dims
        self.num_noise_dims = num_noise_dims
        self.num_dims = num_state_dims + num_noise_dims
        super().__init__(*modules)

    def forward(self, input):
        """
        input = (x, noise)
        """
        assert input.size(-1) == self.num_state_dims + self.num_noise_dims

        return super().forward(input)

    @property
    def global_lipschitz(self):
        """
        Global Lipschitz constant
        :return:
        """
        return None


class Dynamics(torch.nn.Sequential):
    """
    z_{k+1} = dynamics(z_k), with z the state or noise
    """
    num_dims = None

    def __init__(self, *args):
        super().__init__(*args)


class CompositionalStructure:
    """
    z_{k+1} = dynamics[-1] o ... o dynamics[0](z_k), with z_k being x_k, noise_k OR OR (x_k, noise_k)

    For Dynamics or StochasticDynamics of with a CompositionalStructure, global_lbp_sq_norm_fx_fc is applied
    iteratively over the lowest level of the Sequential module.
    """
    pass

class Separable:
    pass


class AdditiveGaussianDynamics(StochasticDynamics): # TODO Why Gaussian? This works for any additive noise
    """
    Special case of StochasticDynamics:
    x_{k+1} = state_dynamics(x_k) + noise_dynamics(noise_k)
    """
    def __init__(self, state_dynamics: Dynamics, noise_dynamics: Optional[Dynamics] = None):
        if noise_dynamics is None:
            noise_dynamics = IdentityDynamics(state_dynamics.num_dims)

        if not state_dynamics.num_dims == noise_dynamics.num_dims:
            raise ValueError("The state and noise dynamics should have the same number of dimensions")

        super().__init__(
            num_state_dims=state_dynamics.num_dims,
            num_noise_dims=noise_dynamics.num_dims,
            modules=[
                bp.Parallel(
                    state_dynamics,
                    noise_dynamics,
                    split_size=state_dynamics.num_dims),
                bp.VectorAdd()
            ])

        self._global_lipschitz = state_dynamics.global_lipschitz

    @property
    def global_lipschitz(self):
        return self._global_lipschitz

    @property
    def state_dynamics(self):
        return self[0].subnetworks[0]

class LinearStochasticDynamics(StochasticDynamics):
    num_state_dims = 3
    num_noise_dims = 3

    def __init__(self, diagonal: Union[torch.Tensor, list], **kwargs):
        assert len(diagonal) == self.num_state_dims + self.num_noise_dims

        super().__init__(
            num_state_dims=self.num_state_dims,
            num_noise_dims=self.num_noise_dims,
            modules=[
                LinearDiagonalDynamics(diagonal),
                bp.VectorAdd()
            ]
        )

    @property
    def global_lipschitz(self):
        return self[0].global_lipschitz

class IdentityDynamics(Dynamics):
    def __init__(self, num_dims: int = 1, **kwargs):
        self.num_dims = num_dims
        super().__init__(lbp.Identity(num_dims))

    @property
    def global_lipschitz(self):
        return 1

class LinearDynamics(Dynamics):
    def __init__(self,
                 weight: Union[torch.Tensor, list],
                 bias: Optional[Union[torch.Tensor, list]] = None,
                 **kwargs):
        if isinstance(weight, list):
            weight = torch.tensor(weight)
        if isinstance(bias, list):
            bias = torch.tensor(bias)

        self.num_dims = weight.size(-1)
        self._global_lipschitz = torch.linalg.svd(weight).S[0]

        super().__init__(lbp.Linear(weight, bias))

    @property
    def global_lipschitz(self):
        return self._global_lipschitz

class LinearDiagonalDynamics(LinearDynamics, Separable):
    def __init__(self,
                 diagonal: Union[torch.Tensor, list],
                 **kwargs):
        if isinstance(diagonal, list):
            diagonal = torch.tensor(diagonal)

        super().__init__(torch.diag(diagonal))

class LinearBoundedDynamics(Dynamics):
    def __init__(self,
                 weight: Union[torch.Tensor, list],
                 bias: Optional[Union[torch.Tensor, list]] = None,
                 lower_bound: Optional[Union[float, torch.Tensor, list]] = -torch.inf,
                 upper_bound: Optional[Union[float, torch.Tensor, list]] = torch.inf,
                 **kwargs):
        if isinstance(weight, list):
            weight = torch.tensor(weight)
        if isinstance(bias, list):
            bias = torch.tensor(bias)

        assert not lower_bound in [None, -torch.inf] or not upper_bound in [None, -torch.inf]

        self.num_dims = weight.size(-1)
        self._global_lipschitz = torch.linalg.svd(weight).S[0]

        super().__init__(lbp.Linear(weight, bias), bp.Clamp(lower_bound, upper_bound))

    @property
    def global_lipschitz(self):
        return self._global_lipschitz

class DiagonalLinearBoundedDynamics(LinearBoundedDynamics, Separable):
    def __init__(self,
                 diagonal: Union[torch.Tensor, list],
                 lower_bound: Union[float, torch.Tensor, list],
                 upper_bound: Union[float, torch.Tensor, list],
                 **kwargs):
        if isinstance(diagonal, list):
            diagonal = torch.tensor(diagonal)

        super().__init__(
            weight=torch.diag(diagonal),
            bias=None,
            lower_bound=lower_bound,
            upper_bound=upper_bound)

class BoundedLinearDynamics(Dynamics):
    def __init__(self,
                 weight: Union[torch.Tensor, list],
                 bias: Optional[Union[torch.Tensor, list]] = None,
                 lower_bound: Optional[Union[float, torch.Tensor, list]] = -torch.inf,
                 upper_bound: Optional[Union[float, torch.Tensor, list]] = torch.inf,
                 **kwargs):
        if isinstance(weight, list):
            weight = torch.tensor(weight)
        if isinstance(bias, list):
            bias = torch.tensor(bias)

        assert not lower_bound in [None, -torch.inf] or not upper_bound in [None, -torch.inf]

        self.num_dims = weight.size(-1)
        self._global_lipschitz = torch.linalg.svd(weight).S[0]

        super(BoundedLinearDynamics, self).__init__(bp.Clamp(lower_bound, upper_bound), lbp.Linear(weight, bias))

    @property
    def global_lipschitz(self):
        return self._global_lipschitz

class PiecewiseAffineBLock(Dynamics):
    def __init__(self, min, max, dynamics):
        min = torch.as_tensor(min) if not torch.is_tensor(min) else min
        max = torch.as_tensor(max) if not torch.is_tensor(max) else max
        super().__init__(
            lbp.BoxedIdentity(min=min, max=max),
            dynamics
        )

    @property
    def global_lipschitz(self):
        return self[1].global_lipschitz


# --- Helpers -----------------------------------------------------------------
class StochasticDynamicsFactoryFn(Protocol):
    def __call__(self, **kwargs) -> StochasticDynamics: ...

def additive(inner_ctor: Callable[..., Dynamics]) -> StochasticDynamicsFactoryFn:
    """
    Turn a plain Dynamics constructor into a factory that returns
    AdditiveGaussianDynamics(inner_ctor(**kwargs), noise_dynamics).
    Accepts an optional `noise_dynamics` in kwargs.
    """
    def factory(**kwargs) -> StochasticDynamics:
        noise = kwargs.pop("noise_dynamics", None)  # allow override
        inner = inner_ctor(**kwargs)
        return AdditiveGaussianDynamics(inner, noise_dynamics=noise)
    return factory


# --- Registry ----------------------------------------------------------------
class GetDynamics:
    _registery = dict(
        LinearDynamics=additive(LinearDynamics),
        LinearBoundedDynamics=additive(LinearBoundedDynamics),
        BoundedLinearDynamics=additive(BoundedLinearDynamics),
        LinearDiagonalDynamics=additive(LinearDiagonalDynamics),
        DiagonalLinearBoundedDynamics=additive(DiagonalLinearBoundedDynamics)
    )

    def __init__(self):
        self._registry: Dict[str, Union[StochasticDynamicsFactoryFn, StochasticDynamics]] = {}

    def register(self, name: str, factory: Union[StochasticDynamicsFactoryFn, StochasticDynamics]) -> None:
        if name in self._registry:
            raise ValueError(f"Factory already registered for '{name}'")
        self._registry[name] = factory

    def __call__(self, dynamics_type: str, **kwargs) -> StochasticDynamics: # TODO change to name
        try:
            return self._registry[dynamics_type](**kwargs)
        except KeyError:
            raise ValueError(f"Unknown dynamics: {dynamics_type}")
        