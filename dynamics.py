import torch
from typing import Union, Optional
import bound_propagation as bp
import math

import linear_bound_propagation as lbp


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


class AdditiveGaussianDynamics(StochasticDynamics):
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


class LinearSigmoidStochasticDynamics(StochasticDynamics, CompositionalStructure):
    num_state_dims = 3
    num_noise_dims = 3

    def __init__(self, diagonal: Union[torch.Tensor, list], **kwargs):
        super().__init__(
            num_state_dims=self.num_state_dims,
            num_noise_dims=self.num_noise_dims,
            modules=[
                LinearStochasticDynamics(diagonal),
                SigmoidDynamics(self.num_state_dims)
            ]
        )

    @property
    def global_lipschitz(self):
        return self[0].global_lipschitz * self[1].global_lipschitz


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


class SigmoidDynamics(Dynamics, Separable):
    def __init__(self, num_dims: int = 1, **kwargs):
        super().__init__(torch.nn.Sigmoid())
        self.num_dims = num_dims

    @property
    def global_lipschitz(self):
        return 0.25


class DiagonalLinearSigmoidDynamics(Dynamics, Separable):
    def __init__(self, diagonal: Union[torch.Tensor, list], **kwargs):
        if isinstance(diagonal, list):
            diagonal = torch.tensor(diagonal)
        self.num_dims = diagonal.size(0)
        self._diagonal = diagonal

        super().__init__(
            LinearDiagonalDynamics(diagonal),
            SigmoidDynamics(self.num_dims)
        )

    @property
    def global_lipschitz(self):
        return self._diagonal.abs().max() * 0.25


class LinearSigmoidDynamics(Dynamics, CompositionalStructure):
    def __init__(self,
                 weight: Union[torch.Tensor, list],
                 bias: Optional[Union[torch.Tensor, list]] = None,
                 **kwargs):
        if isinstance(weight, list):
            weight = torch.tensor(weight)

        self.num_dims = weight.size(-1)

        super().__init__(
            LinearDynamics(weight, bias),
            SigmoidDynamics(self.num_dims)
        )

    @property
    def global_lipschitz(self):
        return self[0].global_lipschitz * self[1].global_lipschitz


class MountainCarDynamics(Dynamics):
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


class DubinsCarDynamics(Dynamics):
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

class PiecewiseAffineBLock(Dynamics):
    def __init__(self, min, max, dynamics):
        super().__init__(
            lbp.BoxedIdentity(min=min, max=max),
            dynamics
        )

    @property
    def global_lipschitz(self):
        return self[1].global_lipschitz

class PiecewiseAffine4modes2dDynamics(Dynamics):
    num_dims = 2
    def __init__(self, **kwargs):
        def mat(theta, rho, delta):
            theta = theta if torch.is_tensor(theta) else torch.as_tensor(theta)
            rho = rho if torch.is_tensor(rho) else torch.as_tensor(rho)
            delta = delta if torch.is_tensor(delta) else torch.as_tensor(delta)
            return rho * torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]) + delta

        theta = -math.pi / 2.
        rho = 0.4
        delta =  0.
        mode1 = PiecewiseAffineBLock(min=-torch.ones(2), max=torch.zeros(2),
                                     dynamics=LinearDynamics(
                                         weight=mat(theta, rho, delta),
                                         bias=torch.zeros(2))
                                     )
        mode2 = PiecewiseAffineBLock(min=torch.tensor([-1., 0.]), max=torch.tensor([0., 1.]),
                                     dynamics=LinearDynamics(
                                         weight=mat(theta, rho, -delta),
                                         bias=torch.zeros(2))
                                     )
        mode3 = PiecewiseAffineBLock(min=torch.zeros(2), max=torch.ones(2),
                                     dynamics=LinearDynamics(
                                         weight=mat(theta, rho, delta),
                                         bias=torch.zeros(2)))
        mode4 = PiecewiseAffineBLock(min=torch.tensor([0., -1.]), max=torch.tensor([1.,0.]),
                                     dynamics=LinearDynamics(
                                         weight=mat(theta, rho, -delta),
                                         bias=torch.zeros(2)))

        super().__init__(
            bp.Clamp(min=-torch.ones(2), max=torch.ones(2)),
            bp.Parallel(mode1, mode2, mode3, mode4),
            bp.VectorAdd(),
            bp.VectorAdd()
            # LinearDynamics(weight=torch.eye(2) * 1.5)
        )

    @property
    def global_lipschitz(self):
        global_lipschitz = []
        for mode in self[1].subnetworks:
            global_lipschitz.append(mode.global_lipschitz)
        return max(global_lipschitz)

class FourModesOpenLoopDynamics(Dynamics):

    def __init__(self, control: int = 1, **kwargs):

        self.num_dims = 2

        linear_part = IdentityDynamics(num_dims=2)

        if control == 1:
            trig_part = torch.nn.Sequential(
                lbp.Linear(
                    torch.tensor([
                        [0.0, 1.0],
                        [1.0, 0.0]
                    ]),
                    torch.tensor([0.0, torch.pi / 2])
                    ),
                bp.Sin(),
                lbp.Linear(
                    torch.tensor([
                        [0.2, 0.0],
                        [0.0, 0.4]
                    ]),
                    torch.tensor([0.5, 0.0])
                ),
            )
        elif control == 2:
            trig_part = torch.nn.Sequential(
                lbp.Linear(
                    torch.tensor([
                        [0.0, 1.0],
                        [1.0, 0.0]
                    ]),
                    torch.tensor([0.0, torch.pi / 2])
                    ),
                bp.Sin(),
                lbp.Linear(
                    torch.tensor([
                        [0.2, 0.0],
                        [0.0, 0.4]
                    ]),
                    torch.tensor([-0.5, 0.0])
                ),
            )
        elif control==3:
            trig_part = torch.nn.Sequential(
                lbp.Linear(
                    torch.tensor([
                        [0.0, 1.0],
                        [1.0, 0.0]
                    ]),
                    torch.tensor([torch.pi / 2, 0.0])
                    ),
                bp.Sin(),
                lbp.Linear(
                    torch.tensor([
                        [0.4, 0.0],
                        [0.0, 0.2]
                    ]),
                    torch.tensor([0.0, 0.5])
                ),
            )
        elif control==4:
            trig_part = torch.nn.Sequential(
                lbp.Linear(
                    torch.tensor([
                        [0.0, 1.0],
                        [1.0, 0.0]
                    ]),
                    torch.tensor([torch.pi / 2, 0.0])
                    ),
                bp.Sin(),
                lbp.Linear(
                    torch.tensor([
                        [0.4, 0.0],
                        [0.0, 0.2]
                    ]),
                    torch.tensor([0.0, -0.5])
                ),
            )
        else:
            raise Exception

        super().__init__(
            bp.Parallel(linear_part, trig_part),
            bp.VectorAdd(),
        )

    @property
    def global_lipschitz(self):
        return 1.4


def get_dynamics(dynamics_type: str, **kwargs):
    if dynamics_type == 'LinearDynamics':
        return AdditiveGaussianDynamics(LinearDynamics(**kwargs))
    elif dynamics_type == 'LinearBoundedDynamics':
        return AdditiveGaussianDynamics(LinearBoundedDynamics(**kwargs))
    elif dynamics_type == 'BoundedLinearDynamics':
        return AdditiveGaussianDynamics(BoundedLinearDynamics(**kwargs))
    elif dynamics_type == 'LinearDiagonalDynamics':
        return AdditiveGaussianDynamics(LinearDiagonalDynamics(**kwargs))
    elif dynamics_type == 'DiagonalLinearBoundedDynamics':
        return AdditiveGaussianDynamics(DiagonalLinearBoundedDynamics(**kwargs))
    elif dynamics_type == 'SigmoidDynamics':
        return AdditiveGaussianDynamics(SigmoidDynamics(**kwargs))
    elif dynamics_type == 'LinearSigmoidDynamics':
        return AdditiveGaussianDynamics(LinearSigmoidDynamics(**kwargs))
    elif dynamics_type == 'DiagonalLinearSigmoidDynamics':
        return AdditiveGaussianDynamics(DiagonalLinearSigmoidDynamics(**kwargs))
    elif dynamics_type == 'MountainCarDynamics':
        return AdditiveGaussianDynamics(MountainCarDynamics(**kwargs))
    elif dynamics_type == 'DubinsCarDynamics':
        return AdditiveGaussianDynamics(DubinsCarDynamics(**kwargs))
    elif dynamics_type == 'FourModesOpenLoopDynamics':
        return AdditiveGaussianDynamics(FourModesOpenLoopDynamics(**kwargs))
    elif dynamics_type == 'PiecewiseAffine4modes2dDynamics':
        return AdditiveGaussianDynamics(PiecewiseAffine4modes2dDynamics(**kwargs))
    elif dynamics_type == 'LinearSigmoidStochasticDynamics':
        return LinearSigmoidStochasticDynamics(**kwargs)
    else:
        raise ValueError(f"Unknown dynamics: {dynamics_type}")