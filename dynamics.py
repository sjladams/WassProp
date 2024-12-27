import torch
from typing import Union, Optional
import bound_propagation as bp
from bound_propagation import Sin
from torch.nn import Sigmoid
from torch import nn

from modules import ScalarMult, ScalarAdd, Linear, Sum


class Dynamics(torch.nn.Sequential):
    num_dims = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def global_lipschitz(self):
        """
        Global Lipschitz constant
        :return:
        """
        return None


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


class NonAdditiveGaussianNoiseDynamics(Dynamics):
    def __init__(self, diagonal: Union[torch.Tensor, list], **kwargs):
        if isinstance(diagonal, list):
            diagonal = torch.tensor(diagonal)
        self.num_dims = diagonal.size(0)
        self._diagonal = diagonal

        super(NonAdditiveGaussianNoiseDynamics, self).__init__(
            LinearDiagonalDynamics(diagonal, min=-torch.inf, max=torch.inf),
            SigmoidDynamics(self.num_dims),
            Sum(self.num_dims)
        )

    @property
    def global_lipschitz(self):
        return self._diagonal.abs().max() * 0.25 * 2 #TODO: CHECK


class LinearPartForMountainCar(Dynamics):
    def __init__(self, action, **kwargs):
        super(LinearPartForMountainCar, self).__init__(
            bp.Clamp(-0.5, 1.2), #TODO: See TODO below, this is a temporary test
            Linear(
                torch.tensor([
                    [1.0, 0.0],
                    [1.0, 1.0]
                ]),
                torch.tensor([0.001 * action, 0.0])
            ),
        )

class TrigonometricPartForMountainCar(Dynamics):
    def __init__(self, **kwargs):
        super(TrigonometricPartForMountainCar, self).__init__(
            Linear(
                torch.tensor([
                    [0.0, 3.0],
                    [0.0, 0.0]
                ]),
                torch.tensor([torch.pi / 2, 0.0])),
            bp.Sin(),
            Linear(
                torch.tensor([
                    [-0.0025, 0.0],
                    [0.0, 0.0]
                ]),
                torch.tensor([0.0, 0.0])),
        )

class MountainCarDynamics(Dynamics):
    def __init__(self,
                 num_dims:int = 2,
                 action:float = 1.0,
                 lower_bound: Optional[Union[float, torch.Tensor, list]] = -torch.inf,
                 upper_bound: Optional[Union[float, torch.Tensor, list]] = torch.inf,
                 **kwargs):
        self.num_dims = num_dims
        self.action = action

        linear_part = LinearPartForMountainCar(self.action)
        trig_part = TrigonometricPartForMountainCar()

        super(MountainCarDynamics, self).__init__(
            bp.Parallel(linear_part, trig_part, split_size=linear_part.num_dims),
            bp.VectorAdd(),
            #bp.Clamp(lower_bound, upper_bound) #TODO: Not working to Clamp here, not working to clamp with tensor
        )

    @property
    def global_lipschitz(self):
        return 2


def get_dynamics(dynamics_type: str, **kwargs):
    if dynamics_type == 'LogisticMap':
        return LogisticMap(**kwargs)
    elif dynamics_type == 'LinearDynamics':
        return LinearDynamics(**kwargs)
    elif dynamics_type == 'LinearBoundedDynamics':
        return LinearBoundedDynamics(**kwargs)
    elif dynamics_type == 'BoundedLinearDynamics':
        return BoundedLinearDynamics(**kwargs)
    elif dynamics_type == 'LinearDiagonalDynamics':
        return LinearDiagonalDynamics(**kwargs)
    elif dynamics_type == 'LinearDiagonalBoundedDynamics':
        return LinearDiagonalBoundedDynamics(**kwargs)
    elif dynamics_type == 'SigmoidDynamics':
        return SigmoidDynamics(**kwargs)
    elif dynamics_type == 'LinearDiagonalSigmoidDynamics':
        return LinearDiagonalSigmoidDynamics(**kwargs)
    elif dynamics_type == 'NonAdditiveGaussianNoiseDynamics':
        return NonAdditiveGaussianNoiseDynamics(**kwargs)
    elif dynamics_type == 'MountainCarDynamics':
        return MountainCarDynamics(**kwargs)
    else:
        raise ValueError(f"Unknown dynamics: {dynamics_type}")