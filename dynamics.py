import torch
from typing import Union, Optional
import bound_propagation as bp

from modules import ScalarMult, ScalarAdd, Linear, Sum


class StochasticDynamics(torch.nn.Sequential):
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


class NonAdditiveGaussianNoiseDynamics(StochasticDynamics):
    def __init__(self, diagonal: Union[torch.Tensor, list], **kwargs):
        num_state_dims=1
        num_noise_dims=1

        super(NonAdditiveGaussianNoiseDynamics, self).__init__(
            num_state_dims=num_state_dims,
            num_noise_dims=num_noise_dims,
            modules=[
                LinearDiagonalDynamics(diagonal, min=-torch.inf, max=torch.inf),
                SigmoidDynamics(num_state_dims+num_noise_dims),
                Sum(num_state_dims+num_noise_dims)
            ]
        )

    @property
    def global_lipschitz(self):
        return self[0].global_lipschitz * 0.25 #TODO: CHECK


class Dynamics(torch.nn.Sequential):
    num_dims = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AdditiveGaussianDynamics(StochasticDynamics):
    def __init__(self, state_dynamics: Dynamics):
        super().__init__(state_dynamics.num_dims, num_noise_dims=state_dynamics.num_dims,
                         modules=[
                             bp.Parallel(
                                 state_dynamics,
                                 torch.nn.Identity(),
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


class LogisticMap(Dynamics):
    num_dims = 1
    def __init__(self,
                 r: float,
                 **kwargs):
        self.r = r

        clamp_0_1 = bp.Sub(torch.nn.ReLU(), torch.nn.Sequential(ScalarAdd(self.num_dims, -1), torch.nn.ReLU()))

        super(LogisticMap, self).__init__(
            bp.Mul(clamp_0_1, torch.nn.Sequential(clamp_0_1, ScalarAdd(self.num_dims, -1))),
            ScalarMult(self.num_dims, -r)
        )

    @property
    def global_lipschitz(self):
        return self.r


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

        super(LinearDynamics, self).__init__(Linear(weight, bias))

    @property
    def global_lipschitz(self):
        return self._global_lipschitz

class LinearDiagonalDynamics(LinearDynamics):
    def __init__(self,
                 diagonal: Union[torch.Tensor, list],
                 **kwargs):
        if isinstance(diagonal, list):
            diagonal = torch.tensor(diagonal)

        super(LinearDiagonalDynamics, self).__init__(torch.diag(diagonal))


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

        super(LinearBoundedDynamics, self).__init__(Linear(weight, bias), bp.Clamp(lower_bound, upper_bound))

    @property
    def global_lipschitz(self):
        return self._global_lipschitz


class LinearDiagonalBoundedDynamics(LinearBoundedDynamics):
    def __init__(self,
                 diagonal: Union[torch.Tensor, list],
                 lower_bound: Union[float, torch.Tensor, list],
                 upper_bound: Union[float, torch.Tensor, list],
                 **kwargs):
        if isinstance(diagonal, list):
            diagonal = torch.tensor(diagonal)

        super(LinearDiagonalBoundedDynamics, self).__init__(weight=torch.diag(diagonal),
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

        super(BoundedLinearDynamics, self).__init__(bp.Clamp(lower_bound, upper_bound), Linear(weight, bias))

    @property
    def global_lipschitz(self):
        return self._global_lipschitz


class SigmoidDynamics(Dynamics):
    def __init__(self, num_dims: int = 1, **kwargs):
        super(SigmoidDynamics, self).__init__(torch.nn.Sigmoid())
        self.num_dims = num_dims

    @property
    def global_lipschitz(self):
        return 0.25


class LinearDiagonalSigmoidDynamics(Dynamics):
    def __init__(self, diagonal: Union[torch.Tensor, list], **kwargs):
        if isinstance(diagonal, list):
            diagonal = torch.tensor(diagonal)
        self.num_dims = diagonal.size(0)
        self._diagonal = diagonal

        super(LinearDiagonalSigmoidDynamics, self).__init__(
            LinearDiagonalDynamics(diagonal, min=-torch.inf, max=torch.inf),
            SigmoidDynamics(self.num_dims)
        )

    @property
    def global_lipschitz(self):
        return self._diagonal.abs().max() * 0.25


class MountainCarDynamics(Dynamics):
    def __init__(self, action: float = 1.0, **kwargs):
        self.num_dims = 2

        linear_part = torch.nn.Sequential(
            bp.Clamp(-0.5, 1.2),
            Linear(
                torch.tensor([
                    [1.0, 0.0],
                    [1.0, 1.0]
                ]),
                torch.tensor([0.001 * action, 0.0])
            )
        )

        trig_part = torch.nn.Sequential(
            Linear(
                torch.tensor([
                    [0.0, 3.0],
                    [0.0, 0.0]
                ]),
                torch.tensor([torch.pi / 2, 0.0])
                ),
            bp.Sin(),
            Linear(
                torch.tensor([
                    [-0.0025, 0.0],
                    [0.0, 0.0]
                ]),
                torch.tensor([0.0, 0.0])
            ),
        )

        super(MountainCarDynamics, self).__init__(
            bp.Parallel(linear_part, trig_part),
            bp.VectorAdd(),
        )

    @property
    def global_lipschitz(self):
        return 2

class DiscreteMountainCarDynamics(StochasticDynamics):
    def __init__(self, action: float = 1.0, **kwargs):
        num_state_dims=2
        num_noise_dims=2

        mountain_car = MountainCarDynamics(action, **kwargs)

        super(DiscreteMountainCarDynamics, self).__init__(
            num_state_dims=num_state_dims,
            num_noise_dims=num_noise_dims,
            modules=[
                bp.Parallel(
                    mountain_car,
                    torch.nn.Identity(),
                    split_size=num_state_dims),
                bp.VectorAdd()
            ]
        )

    @property
    def global_lipschitz(self):
        return 2

class DubinsCarDynamics(Dynamics):
    def __init__(self, velocity: float = 1.0, u: float = 0.5, h: float = 0.3, **kwargs):
        self.num_dims = 3
        self.velocity = velocity
        self.u = u
        self.h = h

        linear_part = Linear(
                torch.tensor([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ]),
                torch.tensor([0.0, 0.0, h * u])
            )

        trig_part = torch.nn.Sequential(
            Linear(
                torch.tensor([
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0]
                ]),
                torch.tensor([torch.pi / 2, 0.0, 0.0])
                ),
            bp.Sin(),
            Linear(
                torch.tensor([
                    [h * velocity, 0.0, 0.0],
                    [0.0, h * velocity, 0.0],
                    [0.0, 0.0, 0.0]
                ]),
                torch.tensor([0.0, 0.0, 0.0])
            ),
        )

        super(DubinsCarDynamics, self).__init__(
            bp.Parallel(linear_part, trig_part),
            bp.VectorAdd(),
        )

    @property
    def global_lipschitz(self):
        return 1 + self.h * self.velocity

def get_dynamics(dynamics_type: str, additive_gaussian_noise: bool = True, **kwargs):
    if dynamics_type == 'LogisticMap' and additive_gaussian_noise:
        return AdditiveGaussianDynamics(LogisticMap(**kwargs))
    elif dynamics_type == 'LinearDynamics' and additive_gaussian_noise:
        return AdditiveGaussianDynamics(LinearDynamics(**kwargs))
    elif dynamics_type == 'LinearBoundedDynamics' and additive_gaussian_noise:
        return AdditiveGaussianDynamics(LinearBoundedDynamics(**kwargs))
    elif dynamics_type == 'BoundedLinearDynamics' and additive_gaussian_noise:
        return AdditiveGaussianDynamics(BoundedLinearDynamics(**kwargs))
    elif dynamics_type == 'LinearDiagonalDynamics' and additive_gaussian_noise:
        return AdditiveGaussianDynamics(LinearDiagonalDynamics(**kwargs))
    elif dynamics_type == 'LinearDiagonalBoundedDynamics' and additive_gaussian_noise:
        return AdditiveGaussianDynamics(LinearDiagonalBoundedDynamics(**kwargs))
    elif dynamics_type == 'SigmoidDynamics' and additive_gaussian_noise:
        return AdditiveGaussianDynamics(SigmoidDynamics(**kwargs))
    elif dynamics_type == 'LinearDiagonalSigmoidDynamics' and additive_gaussian_noise:
        return AdditiveGaussianDynamics(LinearDiagonalSigmoidDynamics(**kwargs))
    elif dynamics_type == 'MountainCarDynamics' and additive_gaussian_noise:
        return AdditiveGaussianDynamics(MountainCarDynamics(**kwargs))
    elif dynamics_type == 'DubinsCarDynamics' and additive_gaussian_noise:
        return AdditiveGaussianDynamics(DubinsCarDynamics(**kwargs))
    elif dynamics_type == 'DiscreteMountainCarDynamics':
        return DiscreteMountainCarDynamics(**kwargs)
    elif dynamics_type == 'NonAdditiveGaussianNoiseDynamics':
        return NonAdditiveGaussianNoiseDynamics(**kwargs)
    else:
        raise ValueError(f"Unknown dynamics: {dynamics_type}")