from typing import Union, Optional, List, Iterable
import os

import torch
from torch import nn

import bound_propagation as bp

from wass_prop import GetStochasticDynamics
from wass_prop.dynamics import Dynamics, Linear, LinearDynamics, PreBoundedDynamics, IndicatorDynamics, StochasticDynamics, LinearStochasticDynamics, additive, AdditiveNoiseDynamics
import utils

DATA_FOLDER = f"{os.path.dirname(os.path.abspath(__file__))}{os.sep}data{os.sep}"

# -- Deterministic Dynamics --------------------------------------------------------------------------------------------
class SigmoidDynamics(Dynamics):
    def __init__(self, num_dims: int = 1):
        self._separable = True
        super().__init__(num_dims=num_dims, modules=torch.nn.Sigmoid())

class TanhDynamics(Dynamics):
    def __init__(self, num_dims: int = 1):
        self._separable = True
        super().__init__(num_dims=num_dims, modules=torch.nn.Tanh())

class BoundedLinearDynamics(PreBoundedDynamics):
    def __init__(
        self,
        weight: Union[torch.Tensor, list],
        lower: Union[float, torch.Tensor, list],
        upper: Union[float, torch.Tensor, list],
        bias: Optional[Union[torch.Tensor, list]] = None
    ):
        super().__init__(
            LinearDynamics(weight, bias),
            lower=lower,
            upper=upper
        )

class LinearSigmoidDynamics(Dynamics):
    def __init__(
        self,
        weight: Union[torch.Tensor, list],
        bias: Optional[Union[torch.Tensor, list]] = None,
    ):
        linear_dynamics = LinearDynamics(weight, bias)
        self._seperable = linear_dynamics.separable
    
        super().__init__(
            num_dims=linear_dynamics.num_dims, 
            modules=[linear_dynamics, SigmoidDynamics(self.num_dims)]
        )

class DiagonalSigmoidDynamics(Dynamics):
    def __init__(
        self, 
        diagonal: Union[torch.Tensor, list]
    ):
        linear_dynamics = LinearDynamics(torch.diag(torch.as_tensor(diagonal)))
        self._seperable = True

        super().__init__(
            num_dims=linear_dynamics.num_dims, 
            modules=[linear_dynamics, SigmoidDynamics(linear_dynamics.num_dims)]
        )

class MountainCarDynamics(Dynamics):
    def __init__(self, action: float = 2.0):
        """
        Actions:
        - 0: Accelerate to the left
        - 1: Don’t accelerate
        - 2: Accelerate to the right

        State is [velocity, position]. 
        """

        velocity_part = bp.Add(
            Linear(
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([0.001 * (action - 1.)]),
            ),
            nn.Sequential(
                Linear(torch.tensor([[0.0, 3.0]])),
                bp.Cos(),
                Linear(torch.tensor([[-0.0025]])),
            ),
        )

        position_part = Linear(torch.tensor([[1.0, 1.0]]))

        super().__init__(
            num_dims=2,
            modules=[
                bp.Parallel(velocity_part, position_part),
                bp.Clamp(torch.tensor([-0.07, -1.2]), torch.tensor([0.07, 0.6])),
            ]
        )


class DubinsCarDynamics(Dynamics):
    def __init__(self, velocity: float = 5.0, u: float = 2.0, h: float = 0.3):
        self.velocity = velocity
        self.u = u
        self.h = h

        linear_part = Linear(torch.eye(3), torch.tensor([0.0, 0.0, h * u]))

        trig_part = nn.Sequential(
            Linear(
                torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
                torch.tensor([torch.pi / 2, 0.0]),
            ),
            bp.Sin(),
            Linear(
                torch.tensor([
                    [h * velocity, 0.0],
                    [0.0, h * velocity],
                    [0.0, 0.0],
                ]),
            ),
        )

        super().__init__(
            num_dims=3,
            modules=[bp.Add(linear_part, trig_part)]
        )

    @property
    def global_lipschitz(self):
        """
        Write d = x - c and b = h * velocity. The step's displacement is

            f(x) - f(c) = (d_xy + g(d_theta), d_theta),
            g(d_theta)  = b * (cos(c_theta + d_theta) - cos(c_theta),
                            sin(c_theta + d_theta) - sin(c_theta)),

        where ||g(d_theta)|| = 2 * b * |sin(d_theta / 2)| exactly - the chord of a circle of
        radius b. Because d_xy ranges freely it can be aligned with g, so

            ||f(x) - f(c)||^2 = (||d_xy|| + 2 * b * |sin(d_theta / 2)|)^2 + d_theta^2

        with equality attained rather than merely bounded. Applying 2|sin(s/2)| <= |s|, tight as
        s -> 0, leaves the operator norm of [[1, b], [0, 1]] acting on (||d_xy||, |d_theta|),
        whose largest singular value is (b + sqrt(b^2 + 4)) / 2. That supremum is approached as
        d -> 0 along ||d_xy|| / |d_theta| = 2 / (b + sqrt(b^2 + 4)), so the constant is exact.

        It does not depend on c: the step is equivariant under translation in (x, y) and under
        rotation in theta, so every center sees the same constant and there is no point-wise
        structure left to exploit.
        """

        radius = self.h * self.velocity
        b = torch.as_tensor(radius)
        return (b + (b.square() + 4.0).sqrt()) / 2.0

class Spiral2dDynamics(Dynamics):
    def __init__(self):
        weight = utils.rot_mat(theta=-torch.pi / 8., rho=0.8, delta=0.)
        bias = (torch.eye(2) - weight) @ torch.tensor([0., 0.])
        super().__init__(
            num_dims=2,
            modules=[LinearDynamics(weight=weight, bias=bias)]
        )

class DoubleSpiral2dDynamics(Dynamics):
    def __init__(self):
        region_left = torch.tensor([[-2., -0.75], [0., 1.25]])
        weight_left = utils.rot_mat(theta=torch.pi / 8., rho=0.8, delta=0.)
        bias_left = (torch.eye(2) - weight_left) @ torch.tensor([-1.25, -1.0])

        region_right = torch.tensor([[0., -0.75], [2., 1.25]])
        weight_right = utils.rot_mat(theta=-torch.pi / 8., rho=0.8, delta=0.)
        bias_right = (torch.eye(2) - weight_right) @ torch.tensor([1.25, -1.0])

        mode_left = IndicatorDynamics(
            lower=region_left[0], 
            upper=region_left[1],
            dynamics=LinearDynamics(weight=weight_left, bias=bias_left)
        )
        mode_right = IndicatorDynamics(
            lower=region_right[0], 
            upper=region_right[1],
            dynamics=LinearDynamics(weight=weight_right, bias=bias_right)
        )

        super().__init__(
            num_dims=2,
            modules=[
                bp.Clamp(min=region_left[0], max=region_right[1]),
                bp.Parallel(mode_left, mode_right),
                bp.VectorAdd()
            ]
        )

    @property
    def global_lipschitz(self):
        global_lipschitz = []
        for mode in self[1].subnetworks:
            global_lipschitz.append(mode.global_lipschitz)
        return max(global_lipschitz)

class NeuralPendulumDynamics(Dynamics):
    def __init__(self, activation: str):
        state_dict = torch.load(f'{DATA_FOLDER}{activation}_model_weights_pendulum.pth', weights_only=True)

        weight_fc1 = state_dict["fc1.weight"]
        bias_fc1 = state_dict["fc1.bias"]
        weight_fc2 = state_dict["fc2.weight"]
        bias_fc2 = state_dict["fc2.bias"]
        weight_fc3 = state_dict["fc3.weight"]
        bias_fc3 = state_dict["fc3.bias"]

        if activation == 'sigmoid':
            ActivationDynamics = SigmoidDynamics
        elif activation == 'tanh':
            ActivationDynamics = TanhDynamics
        else:
            raise NotImplementedError(f"Activation {activation} not implemented.")

        super().__init__(
            num_dims=2, 
            modules=[
                LinearDynamics(weight_fc1, bias_fc1),
                ActivationDynamics(bias_fc1.size(0)),
                LinearDynamics(weight_fc2, bias_fc2),
                ActivationDynamics(bias_fc2.size(0)),
                LinearDynamics(weight_fc3, bias_fc3)
            ]
        )

class FourModesOpenLoopDynamics(Dynamics):
    def __init__(self, control: int = 1):
        linear_part = LinearDynamics(weight=torch.eye(2))

        if control == 1:
            trig_part = torch.nn.Sequential(
                Linear(
                    torch.tensor([
                        [0.0, 1.0],
                        [1.0, 0.0]
                    ]),
                    torch.tensor([0.0, torch.pi / 2])
                    ),
                bp.Sin(),
                Linear(
                    torch.tensor([
                        [0.2, 0.0],
                        [0.0, 0.4]
                    ]),
                    torch.tensor([0.5, 0.0])
                ),
            )
        elif control == 2:
            trig_part = torch.nn.Sequential(
                Linear(
                    torch.tensor([
                        [0.0, 1.0],
                        [1.0, 0.0]
                    ]),
                    torch.tensor([0.0, torch.pi / 2])
                    ),
                bp.Sin(),
                Linear(
                    torch.tensor([
                        [0.2, 0.0],
                        [0.0, 0.4]
                    ]),
                    torch.tensor([-0.5, 0.0])
                ),
            )
        elif control==3:
            trig_part = torch.nn.Sequential(
                Linear(
                    torch.tensor([
                        [0.0, 1.0],
                        [1.0, 0.0]
                    ]),
                    torch.tensor([torch.pi / 2, 0.0])
                    ),
                bp.Sin(),
                Linear(
                    torch.tensor([
                        [0.4, 0.0],
                        [0.0, 0.2]
                    ]),
                    torch.tensor([0.0, 0.5])
                ),
            )
        elif control==4:
            trig_part = torch.nn.Sequential(
                Linear(
                    torch.tensor([
                        [0.0, 1.0],
                        [1.0, 0.0]
                    ]),
                    torch.tensor([torch.pi / 2, 0.0])
                    ),
                bp.Sin(),
                Linear(
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
            num_dims=2, 
            modules=[bp.Parallel(linear_part, trig_part), bp.VectorAdd()],
        )

    @property
    def global_lipschitz(self):
        return 1.4

class SwitchedLinearDynamics(Dynamics):
    def __init__(self):
        region = [[-2., -2.], [2., 2.]]

        mat1 = [[0.79, 0.035], [0., 0.825]]
        mat2 = [[0.79, 0.175], [0., 0.825]]
        mat3 = [[0.79, 0.], [0.175, 0.825]]
        mat4 = [[1., 0.2], [-0.2, 1.]]
        mat5 = [[1., -0.2], [0.2, 1.]]
        redun_mat = torch.eye(2)

        mid_region = [[-0.8, -0.8],[0.8, 0.8]]
        mid_block = IndicatorDynamics(lower=mid_region[0], upper=mid_region[1], dynamics=LinearDynamics(weight=redun_mat))

        obs_right = IndicatorDynamics(lower=[1., 1.], upper=[2., 2.], dynamics=LinearDynamics(weight=redun_mat))
        mode2_right = IndicatorDynamics(lower=[mid_region[1][0], 0.25], upper=[2., 1.], dynamics=LinearDynamics(weight=mat2))
        mode5_right = IndicatorDynamics(lower=[mid_region[1][0], -1.], upper=[2., 0.25], dynamics=LinearDynamics(weight=mat5))
        mode1_bottom = IndicatorDynamics(lower=[0., -2.], upper=[2., -1.8], dynamics=LinearDynamics(weight=mat1))
        mode4_bottom = IndicatorDynamics(lower=[0., -1.8], upper=[2., -1.], dynamics=LinearDynamics(weight=mat4))
        mode3 = IndicatorDynamics(lower=[0.3, mid_region[1][1]], upper=[1., 2.], dynamics=LinearDynamics(weight=mat1))
        mode2_bottom = IndicatorDynamics(lower=[-0.6, -2.], upper=[0., mid_region[0][1]], dynamics=LinearDynamics(weight=mat2))
        mode1_bottom_left = IndicatorDynamics(lower=[-1., -2.], upper=[-0.6, mid_region[0][1]], dynamics=LinearDynamics(weight=mat1))

        mode4_top = IndicatorDynamics(lower=[-1.8, 1.], upper=[0.3, 1.8], dynamics=LinearDynamics(weight=mat4))
        mode2_top = IndicatorDynamics(lower=[-2, 1.8], upper=[0.3, 2.], dynamics=LinearDynamics(weight=mat2))
        mode1_left = IndicatorDynamics(lower=[-2., 0.], upper=[-1.8, 1.8], dynamics=LinearDynamics(weight=mat1))
        mode5_left = IndicatorDynamics(lower=[-1.8, 0.], upper=[mid_region[0][0], 1.], dynamics=LinearDynamics(weight=mat5))
        mode2_left = IndicatorDynamics(lower=[-2., -1.], upper=[mid_region[0][0], 0.], dynamics=LinearDynamics(weight=mat2))
        obs_left = IndicatorDynamics(lower=[-2., -2.], upper=[-1., -1.], dynamics=LinearDynamics(weight=redun_mat))

        redun_mode = IndicatorDynamics(lower=region[0], upper=region[1], dynamics=LinearDynamics(weight=torch.zeros((2,2))))

        super().__init__(
            num_dims=2, 
            modules=[
                bp.Clamp(min=torch.as_tensor(region[0]), max=torch.as_tensor(region[1])),
                bp.Parallel(
                    obs_right, mode2_right, mode5_right, mode1_bottom,
                    mode4_bottom, mode3,
                    mode2_bottom, mode1_bottom_left, mode4_top, mode2_top,
                    mid_block,
                    mode1_left, mode5_left, mode2_left, obs_left,
                    redun_mode
                ),
                bp.VectorAdd(), bp.VectorAdd(), bp.VectorAdd(), bp.VectorAdd()
            ]
        )

    @property
    def global_lipschitz(self):
        global_lipschitz = []
        for mode in self[1].subnetworks:
            global_lipschitz.append(mode.global_lipschitz)
        return max(global_lipschitz)

# -- Stochastic Dynamics -----------------------------------------------------------------------------------------------
class LinearSigmoidStochasticDynamics(StochasticDynamics):
    def __init__(
        self, 
        weight: Union[torch.Tensor, list],
        bias: Optional[Union[torch.Tensor, list]] = None,
    ):
        linear_dynamics = LinearStochasticDynamics(weight, bias)
        super().__init__(
            num_state_dims=linear_dynamics.num_state_dims,
            num_noise_dims=linear_dynamics.num_noise_dims,
            modules=[
                linear_dynamics,
                SigmoidDynamics(self.num_state_dims)
            ]
        )


get_stoch_dynamics = GetStochasticDynamics()
get_stoch_dynamics.register('SigmoidDynamics', additive(SigmoidDynamics))
get_stoch_dynamics.register('TanhDynamics', additive(TanhDynamics))
get_stoch_dynamics.register('BoundedLinearDynamics', additive(BoundedLinearDynamics))
get_stoch_dynamics.register('LinearSigmoidDynamics', additive(LinearSigmoidDynamics))
get_stoch_dynamics.register('DiagonalSigmoidDynamics', additive(DiagonalSigmoidDynamics))
get_stoch_dynamics.register('MountainCarDynamics', additive(MountainCarDynamics))
get_stoch_dynamics.register('DubinsCarDynamics', additive(DubinsCarDynamics))
get_stoch_dynamics.register('Spiral2dDynamics', additive(Spiral2dDynamics))
get_stoch_dynamics.register('DoubleSpiral2dDynamics', additive(DoubleSpiral2dDynamics))
get_stoch_dynamics.register('NeuralPendulumDynamics', additive(NeuralPendulumDynamics))
get_stoch_dynamics.register('FourModesOpenLoopDynamics', additive(FourModesOpenLoopDynamics))
get_stoch_dynamics.register('SwitchedLinearDynamics', additive(SwitchedLinearDynamics))

get_stoch_dynamics.register('LinearSigmoidStochasticDynamics', LinearSigmoidStochasticDynamics)
