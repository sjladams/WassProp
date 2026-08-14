from typing import Callable, Dict, Protocol, Optional, Union, List

import torch
import bound_propagation as bp
import pointwise_lipschitz as pl

pl_factory = pl.BoundModelFactory()

from . import utils

class Dynamics(torch.nn.Sequential):

    """
    z_{k+1} = dynamics(z_k), with z the state or noise

    """
    def __init__(self, num_dims: int, modules: Union[List[torch.nn.Module], torch.nn.Module]):
        self.num_dims = num_dims
        super().__init__(*(modules if isinstance(modules, list) else [modules]))
    
    @property
    def global_lipschitz(self) -> Union[float, torch.Tensor]:
        return pl_factory.build(self).lipschitz()

class StochasticDynamics(torch.nn.Sequential):
    """
    x_{k+1} = stochastic_dynamics(x_k, noise_k)
    """
    def __init__(self, num_state_dims: int, num_noise_dims: int, modules: List[torch.nn.Module]):
        self.num_state_dims = num_state_dims
        self.num_noise_dims = num_noise_dims
        self.num_dims = num_state_dims + num_noise_dims
        super().__init__(*modules)

    def forward(self, input: torch.Tensor):
        """
        input = (x, noise)
        """
        assert input.size(-1) == self.num_state_dims + self.num_noise_dims

        return super().forward(input)

    @property
    def global_lipschitz(self) -> Union[float, torch.Tensor]:
        return pl_factory.build(self).lipschitz()

class AdditiveNoiseDynamics(StochasticDynamics):
    """
    Special case of StochasticDynamics:
    x_{k+1} = state_dynamics(x_k) + noise_dynamics(noise_k)
    """
    def __init__(
        self, 
        state_dynamics: Dynamics, 
        noise_dynamics: Optional[Dynamics] = None
    ):
        if noise_dynamics is None:
            noise_dynamics = LinearDynamics(weight=torch.eye(state_dynamics.num_dims))

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

    @property
    def state_dynamics(self):
        return self[0].subnetworks[0]

    @property
    def global_lipschitz(self):
        return self.state_dynamics.global_lipschitz

 
# -- Standard Modules  -------------------------------------------------------------------------------------------------
class Linear(torch.nn.Linear):
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(weight.size(-1), weight.size(-2), bias=bias is not None)
        with torch.no_grad():
            self.weight.copy_(weight)
            if bias is not None:
                self.bias.copy_(bias)

class LinearDynamics(Dynamics):
    def __init__(
        self,
        weight: Union[torch.Tensor, list],
        bias: Optional[Union[torch.Tensor, list]] = None,
    ):
        weight = torch.as_tensor(weight)
        if isinstance(bias, list):
            bias = torch.as_tensor(bias)

        super().__init__(num_dims=weight.size(-1), modules=Linear(weight, bias))

class PreBoundedDynamics(Dynamics):
    def __init__(
        self,
        dynamics: Dynamics,
        lower: Union[float, torch.Tensor, list],
        upper: Union[float, torch.Tensor, list],
    ):
        super().__init__(
            num_dims=dynamics.num_dims, 
            modules=[bp.Clamp(torch.as_tensor(lower), torch.as_tensor(upper)), dynamics]
        )

class PostBoundedDynamics(Dynamics):
    def __init__(
        self,
        dynamics: Dynamics,
        lower: Union[float, torch.Tensor, list],
        upper: Union[float, torch.Tensor, list],
    ):
        super().__init__(
            num_dims=dynamics.num_dims, 
            modules=[dynamics, bp.Clamp(torch.as_tensor(lower), torch.as_tensor(upper))]
        )


class NNLayerDynamics(Dynamics):
    def __init__(
        self,
        weight: Union[torch.Tensor, list],
        bias: Optional[Union[torch.Tensor, list]] = None,
    ):
        weight = torch.as_tensor(weight)
        if isinstance(bias, list):
            bias = torch.as_tensor(bias)

        super().__init__(
            num_dims=weight.size(-1),
            modules=[Linear(weight, bias), torch.nn.Sigmoid()]
        )

class LinearStochasticDynamics(StochasticDynamics):
    def __init__(
        self, 
        weight: Union[torch.Tensor, list], 
        bias: Optional[Union[torch.Tensor, list]] = None
    ):
        weight = torch.as_tensor(weight)

        if isinstance(bias, list):
            bias = torch.as_tensor(bias)
        
        if not weight.ndim == 2:
            raise ValueError("Weight should be a 2D matrix")
        
        if bias is not None and not bias.ndim == 1:
            raise ValueError("Bias should be a 1D vector")

        super().__init__(
            num_state_dims=weight.size(0),
            num_noise_dims=weight.size(1) - weight.size(0),
            modules=[LinearDynamics(weight=weight, bias=bias)]
        )


# --- Helpers -----------------------------------------------------------------
class StochasticDynamicsFactoryFn(Protocol):
    def __call__(self, **kwargs) -> StochasticDynamics: ...

def additive(inner_ctor: Callable[..., Dynamics]) -> StochasticDynamicsFactoryFn:
    """
    Turn a plain Dynamics constructor into a factory that returns
    AdditiveNoiseDynamics(inner_ctor(**kwargs), noise_dynamics).
    Accepts an optional `noise_dynamics` in kwargs.
    """
    def factory(**kwargs) -> StochasticDynamics:
        noise = kwargs.pop("noise_dynamics", None)  # allow override
        inner = inner_ctor(**kwargs)
        return AdditiveNoiseDynamics(inner, noise_dynamics=noise)
    return factory


# --- Registry ----------------------------------------------------------------
class GetStochasticDynamics:
    _registry = dict(
        LinearDynamics=additive(LinearDynamics),
        PreBoundedDynamics=additive(PreBoundedDynamics),
        PostBoundedDynamics=additive(PostBoundedDynamics), 
    )

    def register(self, name: str, factory: Union[StochasticDynamicsFactoryFn, StochasticDynamics]) -> None:
        if name in self._registry:
            raise ValueError(f"Factory already registered for '{name}'")
        self._registry[name] = factory

    def __call__(self, name: str, **kwargs) -> StochasticDynamics:
        try:
            return self._registry[name](**kwargs)
        except KeyError:
            raise ValueError(f"Unknown dynamics: {name}")
        