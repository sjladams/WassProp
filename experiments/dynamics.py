from typing import Union, Optional
import torch
import bound_propagation as bp

from duq_via_wasserstein import GetDynamics
import duq_via_wasserstein.dynamics as dyn
import duq_via_wasserstein.linear_bound_propagation as lbp


class SigmoidDynamics(dyn.Dynamics, dyn.Separable):
    def __init__(self, num_dims: int = 1, **kwargs):
        self.num_dims = num_dims
        super().__init__(lbp.Identity(num_dims))

    @property
    def global_lipschitz(self):
        return 0.25

class TanhDynamics(dyn.Dynamics, dyn.Separable):
    def __init__(self, num_dims: int = 1, **kwargs):
        super().__init__(torch.nn.Tanh())
        self.num_dims = num_dims

    @property
    def global_lipschitz(self):
        return 1.0
    
class LinearSigmoidStochasticDynamics(dyn.StochasticDynamics, dyn.CompositionalStructure):
    num_state_dims = 3
    num_noise_dims = 3

    def __init__(self, diagonal: Union[torch.Tensor, list], **kwargs):
        super().__init__(
            num_state_dims=self.num_state_dims,
            num_noise_dims=self.num_noise_dims,
            modules=[
                dyn.LinearStochasticDynamics(diagonal),
                SigmoidDynamics(self.num_state_dims)
            ]
        )

    @property
    def global_lipschitz(self):
        return self[0].global_lipschitz * self[1].global_lipschitz


class DiagonalLinearSigmoidDynamics(dyn.Dynamics, dyn.Separable):
    def __init__(self, diagonal: Union[torch.Tensor, list], **kwargs):
        if isinstance(diagonal, list):
            diagonal = torch.tensor(diagonal)
        self.num_dims = diagonal.size(0)
        self._diagonal = diagonal

        super().__init__(
            dyn.LinearDiagonalDynamics(diagonal),
            SigmoidDynamics(self.num_dims)
        )

    @property
    def global_lipschitz(self):
        return self._diagonal.abs().max() * 0.25


class LinearSigmoidDynamics(dyn.Dynamics, dyn.CompositionalStructure):
    def __init__(self,
                 weight: Union[torch.Tensor, list],
                 bias: Optional[Union[torch.Tensor, list]] = None,
                 **kwargs):
        if isinstance(weight, list):
            weight = torch.tensor(weight)

        self.num_dims = weight.size(-1)

        super().__init__(
            dyn.LinearDynamics(weight, bias),
            SigmoidDynamics(self.num_dims)
        )

    @property
    def global_lipschitz(self):
        return self[0].global_lipschitz * self[1].global_lipschitz


class MountainCarDynamics(dyn.Dynamics):
    num_dims = 2

    def __init__(self, action: float = 1.0, **kwargs):
        linear_part = torch.nn.Sequential(
            bp.Clamp(-0.5, 1.2),
            lbp.Linear(
                torch.tensor([
                    [1.0, 0.0],
                    [1.0, 1.0]
                ]),
                torch.tensor([0.001 * action, 0.0])
            )
        )

        trig_part = torch.nn.Sequential(
            lbp.Linear(
                torch.tensor([
                    [0.0, 3.0],
                    [0.0, 0.0]
                ]),
                torch.tensor([torch.pi / 2, 0.0])
                ),
            bp.Sin(),
            lbp.Linear(
                torch.tensor([
                    [-0.0025, 0.0],
                    [0.0, 0.0]
                ]),
                torch.tensor([0.0, 0.0])
            ),
        )

        super().__init__(
            torch.nn.Sequential(bp.Parallel(linear_part, trig_part)),
            bp.VectorAdd(),
        )

    @property
    def global_lipschitz(self):
        return 2


class DubinsCarDynamics(dyn.Dynamics):
    num_dims = 3

    def __init__(self, velocity: float = 5.0, u: float = 2.0, h: float = 0.3, **kwargs):
        self.velocity = velocity
        self.u = u
        self.h = h

        linear_part = lbp.Linear(
                torch.tensor([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ]),
                torch.tensor([0.0, 0.0, h * u])
            )

        trig_part = torch.nn.Sequential(
            lbp.Linear(
                torch.tensor([
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0]
                ]),
                torch.tensor([torch.pi / 2, 0.0, 0.0])
                ),
            bp.Sin(),
            lbp.Linear(
                torch.tensor([
                    [h * velocity, 0.0, 0.0],
                    [0.0, h * velocity, 0.0],
                    [0.0, 0.0, 0.0]
                ]),
                torch.tensor([0.0, 0.0, 0.0])
            ),
        )

        super().__init__(
            bp.Parallel(linear_part, trig_part),
            bp.VectorAdd(),
        )

    @property
    def global_lipschitz(self):
        return 1 + self.h * self.velocity

get_dynamics = GetDynamics()
        
get_dynamics.register('SigmoidDynamics', dyn.additive(SigmoidDynamics))
get_dynamics.register('LinearSigmoidDynamics', dyn.additive(LinearSigmoidDynamics))
get_dynamics.register('DiagonalLinearSigmoidDynamics', dyn.additive(DiagonalLinearSigmoidDynamics))
get_dynamics.register('MountainCarDynamics', dyn.additive(MountainCarDynamics))
get_dynamics.register('DubinsCarDynamics', dyn.additive(DubinsCarDynamics))
get_dynamics.register('LinearSigmoidStochasticDynamics', LinearSigmoidStochasticDynamics)
