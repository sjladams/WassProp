import torch
from typing import Union, Optional
import bound_propagation as bp
import math
import os

from sympy.solvers.solveset import NonlinearError

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


def rot_mat(theta, rho, delta):
    theta = theta if torch.is_tensor(theta) else torch.as_tensor(theta)
    rho = rho if torch.is_tensor(rho) else torch.as_tensor(rho)
    delta = delta if torch.is_tensor(delta) else torch.as_tensor(delta)
    return rho * torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]) + delta


class Spiral2dDynamics(Dynamics):
    num_dims = 2
    def __init__(self, **kwargs):
        weight = rot_mat(theta=-math.pi / 8., rho=0.8, delta=0.)
        bias = (torch.eye(2) - weight) @ torch.tensor([0., 0.])
        super().__init__(LinearDynamics(weight=weight, bias=bias))

    @property
    def global_lipschitz(self):
        return self[0].global_lipschitz


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


class DoubleSpiral2dDynamics(Dynamics):
    num_dims = 2
    def __init__(self, **kwargs):
        region = [[-2., -0.75], [2., 1.25]]

        weight_left = rot_mat(theta=math.pi / 8., rho=0.8, delta=0.)
        weight_right = rot_mat(theta=-math.pi / 8., rho=0.8, delta=0.)
        bias_left = (torch.eye(2) - weight_left) @ torch.tensor([-1.25, -1.0])
        bias_right = (torch.eye(2) - weight_right) @ torch.tensor([1.25, -1.0])

        mode_left = PiecewiseAffineBLock(min=torch.tensor(region[0]), max=torch.tensor([0., region[1][1]]),
                                         dynamics=LinearDynamics(weight=weight_left, bias=bias_left))
        mode_right = PiecewiseAffineBLock(min=torch.tensor([0., region[0][1]]), max=torch.tensor(region[1]),
                                          dynamics=LinearDynamics(weight=weight_right, bias=bias_right))

        super().__init__(
            bp.Clamp(min=torch.tensor(region[0]), max=torch.tensor(region[1])),
            bp.Parallel(mode_left, mode_right),
            bp.VectorAdd()
        )

    @property
    def global_lipschitz(self):
        global_lipschitz = []
        for mode in self[1].subnetworks:
            global_lipschitz.append(mode.global_lipschitz)
        return max(global_lipschitz)


class SwitchedLinearDynamics(Dynamics): # \todo change name
    num_dims = 2
    def __init__(self, **kwargs):
        region = [[-2., -2.], [2., 2.]]

        mat1 = [[0.79, 0.035], [0., 0.825]]
        mat2 = [[0.79, 0.175], [0., 0.825]]
        mat3 = [[0.79, 0.], [0.175, 0.825]]
        mat4 = [[1., 0.2], [-0.2, 1.]]
        mat5 = [[1., -0.2], [0.2, 1.]]
        redun_mat = torch.eye(2)

        mid_block = PiecewiseAffineBLock(min=[-1., -1.], max=[1., 1.], dynamics=LinearDynamics(weight=redun_mat))

        obs_right = PiecewiseAffineBLock(min=[1., 1.], max=[2., 2.], dynamics=LinearDynamics(weight=redun_mat))
        mode2_right = PiecewiseAffineBLock(min=[1., 0.], max=[2., 1.], dynamics=LinearDynamics(weight=mat2))
        mode5_right = PiecewiseAffineBLock(min=[1., -1.8], max=[2., 0.], dynamics=LinearDynamics(weight=mat5))
        mode1_bottom = PiecewiseAffineBLock(min=[0., -2.], max=[2., -1.8], dynamics=LinearDynamics(weight=mat1))
        mode4_bottom = PiecewiseAffineBLock(min=[0., -1.8], max=[1., -1.], dynamics=LinearDynamics(weight=mat4))
        mode3 = PiecewiseAffineBLock(min=[0., 1.], max=[1., 2.], dynamics=LinearDynamics(weight=mat3))
        mode2_bottom = PiecewiseAffineBLock(min=[-1., -2.], max=[0., -1.], dynamics=LinearDynamics(weight=mat2))
        mode4_top = PiecewiseAffineBLock(min=[-1.8, 1.], max=[0., 1.8], dynamics=LinearDynamics(weight=mat4))
        mode2_top = PiecewiseAffineBLock(min=[-2, 1.8], max=[0., 2.], dynamics=LinearDynamics(weight=mat2))
        mode1_left = PiecewiseAffineBLock(min=[-2., 0.], max=[-1.8, 1.8], dynamics=LinearDynamics(weight=mat1))
        mode5_left = PiecewiseAffineBLock(min=[-1.8, 0.], max=[-1., 1.], dynamics=LinearDynamics(weight=mat5))
        mode2_left = PiecewiseAffineBLock(min=[-2., -1.], max=[-1., 0.], dynamics=LinearDynamics(weight=mat2))
        obs_left = PiecewiseAffineBLock(min=[-2., -2.], max=[-1., -1.], dynamics=LinearDynamics(weight=redun_mat))

        redun_mode = PiecewiseAffineBLock(min=region[0], max=region[1], dynamics=LinearDynamics(weight=torch.zeros((2,2))))

        super().__init__(
            bp.Clamp(min=torch.as_tensor(region[0]), max=torch.as_tensor(region[1])),
            bp.Parallel(
                obs_right, mode2_right, mode5_right, mode1_bottom,
                mode4_bottom, mode3,
                mode2_bottom, mode4_top, mode2_top,
                mid_block,
                mode1_left, mode5_left, mode2_left, obs_left,
                redun_mode,
                redun_mode
            ),
            bp.VectorAdd(), bp.VectorAdd(), bp.VectorAdd(), bp.VectorAdd()
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

class NeuralPendulumDynamics(Dynamics, CompositionalStructure):

    def __init__(self, **kwargs):
        self.num_dims = 2

        state_dict = torch.load(f'{os.getcwd()}{os.sep}data{os.sep}model_weights_pendulum.pth')

        weight_fc1 = state_dict["fc1.weight"]
        bias_fc1 = state_dict["fc1.bias"]
        weight_fc2 = state_dict["fc2.weight"]
        bias_fc2 = state_dict["fc2.bias"]
        weight_fc3 = state_dict["fc3.weight"]
        bias_fc3 = state_dict["fc3.bias"]

        super().__init__(
            LinearDynamics(weight_fc1, bias_fc1),
            SigmoidDynamics(bias_fc1.size(0)),
            LinearDynamics(weight_fc2, bias_fc2),
            SigmoidDynamics(bias_fc2.size(0)),
            LinearDynamics(weight_fc3, bias_fc3)
        )

    @property
    def global_lipschitz(self):
        return torch.tensor([module.global_lipschitz for module in self]).prod()


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
    elif dynamics_type == 'SwitchedLinearDynamics':
        return AdditiveGaussianDynamics(SwitchedLinearDynamics(**kwargs))
    elif dynamics_type == 'Spiral2dDynamics':
        return AdditiveGaussianDynamics(Spiral2dDynamics(**kwargs))
    elif dynamics_type == 'DoubleSpiral2dDynamics':
        return AdditiveGaussianDynamics(DoubleSpiral2dDynamics(**kwargs))
    elif dynamics_type == 'NeuralPendulumDynamics':
        return AdditiveGaussianDynamics(NeuralPendulumDynamics(**kwargs))
    elif dynamics_type == 'LinearSigmoidStochasticDynamics':
        return LinearSigmoidStochasticDynamics(**kwargs)
    else:
        raise ValueError(f"Unknown dynamics: {dynamics_type}")