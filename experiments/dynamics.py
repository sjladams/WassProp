from typing import Union, Optional, List, Iterable
import torch
import bound_propagation as bp

from duq_via_wasserstein import GetStochasticDynamics
import duq_via_wasserstein.dynamics as dyn


# -- Deterministic Dynamics --------------------------------------------------------------------------------------------
class SigmoidDynamics(dyn.Dynamics):
    def __init__(self, num_dims: int = 1):
        self._separable = True
        super().__init__(num_dims=num_dims, modules=torch.nn.Sigmoid())

    @property
    def global_lipschitz(self):
        return 0.25

class TanhDynamics(dyn.Dynamics):
    def __init__(self, num_dims: int = 1):
        self._separable = True
        super().__init__(num_dims=num_dims, modules=torch.nn.Tanh())

    @property
    def global_lipschitz(self):
        return 1.0

class BoundedLinearDynamics(dyn.PreBoundedDynamics):
    def __init__(
        self,
        weight: Union[torch.Tensor, list],
        lower: Union[float, torch.Tensor, list],
        upper: Union[float, torch.Tensor, list],
        bias: Optional[Union[torch.Tensor, list]] = None
    ):
        super().__init__(
            dyn.LinearDynamics(weight, bias),
            lower=lower,
            upper=upper
        )

class LinearSigmoidDynamics(dyn.Dynamics):
    def __init__(
        self,
        weight: Union[torch.Tensor, list],
        bias: Optional[Union[torch.Tensor, list]] = None,
    ):
        linear_dynamics = dyn.LinearDynamics(weight, bias)
        self._seperable = linear_dynamics.separable
    
        super().__init__(
            num_dims=linear_dynamics.num_dims, 
            modules=[linear_dynamics, SigmoidDynamics(self.num_dims)]
        )

    @property
    def global_lipschitz(self):
        return torch.tensor([module.global_lipschitz for module in self]).prod()

class DiagonalSigmoidDynamics(dyn.Dynamics):
    def __init__(
        self, 
        diagonal: Union[torch.Tensor, list]
    ):
        linear_dynamics = dyn.LinearDynamics(torch.diag(torch.as_tensor(diagonal)))
        self._seperable = True

        super().__init__(
            num_dims=linear_dynamics.num_dims, 
            modules=[linear_dynamics, SigmoidDynamics(linear_dynamics.num_dims)]
        )

    @property
    def global_lipschitz(self):
        return torch.tensor([module.global_lipschitz for module in self]).prod()

class MountainCarDynamics(dyn.Dynamics):
    def __init__(self, action: float = 1.0):
        linear_part = torch.nn.Sequential(
            bp.Clamp(-0.5, 1.2),
            dyn.Linear(
                torch.tensor([
                    [1.0, 0.0],
                    [1.0, 1.0]
                ]),
                torch.tensor([0.001 * action, 0.0])
            )
        )

        trig_part = torch.nn.Sequential(
            dyn.Linear(
                torch.tensor([
                    [0.0, 3.0],
                    [0.0, 0.0]
                ]),
                torch.tensor([torch.pi / 2, 0.0])
                ),
            bp.Sin(),
            dyn.Linear(
                torch.tensor([
                    [-0.0025, 0.0],
                    [0.0, 0.0]
                ]),
                torch.tensor([0.0, 0.0])
            ),
        )

        super().__init__(
            num_dims=2, 
            modules=[
                torch.nn.Sequential(bp.Parallel(linear_part, trig_part)),
                bp.VectorAdd()
            ]
        )

    @property
    def global_lipschitz(self):
        return 2

class DubinsCarDynamics(dyn.Dynamics):
    def __init__(self, velocity: float = 5.0, u: float = 2.0, h: float = 0.3):
        self.velocity = velocity
        self.u = u
        self.h = h

        linear_part = dyn.Linear(
                torch.tensor([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ]),
                torch.tensor([0.0, 0.0, h * u])
            )

        trig_part = torch.nn.Sequential(
            dyn.Linear(
                torch.tensor([
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0]
                ]),
                torch.tensor([torch.pi / 2, 0.0, 0.0])
                ),
            bp.Sin(),
            dyn.Linear(
                torch.tensor([
                    [h * velocity, 0.0, 0.0],
                    [0.0, h * velocity, 0.0],
                    [0.0, 0.0, 0.0]
                ]),
                torch.tensor([0.0, 0.0, 0.0])
            ),
        )

        super().__init__(
            num_dims=3,
            modules=[bp.Parallel(linear_part, trig_part), bp.VectorAdd()]
        )

    @property
    def global_lipschitz(self):
        return 1 + self.h * self.velocity

# -- Stochastic Dynamics -----------------------------------------------------------------------------------------------
class LinearSigmoidStochasticDynamics(dyn.StochasticDynamics):
    def __init__(
        self, 
        weight: Union[torch.Tensor, list],
        bias: Optional[Union[torch.Tensor, list]] = None,
    ):
        linear_dynamics = dyn.LinearStochasticDynamics(weight, bias)
        super().__init__(
            num_state_dims=linear_dynamics.num_state_dims,
            num_noise_dims=linear_dynamics.num_noise_dims,
            modules=[
                linear_dynamics,
                SigmoidDynamics(self.num_state_dims)
            ]
        )

    @property
    def global_lipschitz(self):
        return global_lipschitz_sequential(self.modules)

get_stoch_dynamics = GetStochasticDynamics()
get_stoch_dynamics.register('SigmoidDynamics', dyn.additive(SigmoidDynamics))
get_stoch_dynamics.register('TanhDynamics', dyn.additive(TanhDynamics))
get_stoch_dynamics.register('BoundedLinearDynamics', dyn.additive(BoundedLinearDynamics))
get_stoch_dynamics.register('LinearSigmoidDynamics', dyn.additive(LinearSigmoidDynamics))
get_stoch_dynamics.register('DiagonalSigmoidDynamics', dyn.additive(DiagonalSigmoidDynamics))
get_stoch_dynamics.register('MountainCarDynamics', dyn.additive(MountainCarDynamics))
get_stoch_dynamics.register('DubinsCarDynamics', dyn.additive(DubinsCarDynamics))
get_stoch_dynamics.register('LinearSigmoidStochasticDynamics', LinearSigmoidStochasticDynamics)
