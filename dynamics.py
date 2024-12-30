import torch
from typing import Union, Optional
import bound_propagation as bp

from modules import ScalarMult, ScalarAdd, Linear


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


class MountainCarDynamics(Dynamics):
    def __init__(self, acc=0.9, g=9.82, **kwargs):
        self.acc = acc
        self.g = g

        super(MountainCarDynamics, self).__init__( # Input: p, v
            bp.Cat(
                torch.nn.Sequential(
                    bp.Select([0]),
                    bp.FixedLinear(torch.tensor([[3.0]])),
                    bp.Sin()
                )
            ), # p, v, cos(3p)
            bp.FixedLinear(
                torch.tensor([
                    [1.0, 1.0, 0.0], # p_{t+1}
                    [0.0, 1.0, -self.g]
                ]),
                bias=torch.tensor([0.0, self.acc])
            ), # p + v, v - g cos(3p) + a
            bp.Clamp(
                min=torch.tensor([-10.0, -1.0]),
                max=torch.tensor([10.0, 1.0])
            )
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
    elif dynamics_type == 'MountainCarDynamics':
        return MountainCarDynamics(**kwargs)
    else:
        raise ValueError(f"Unknown dynamics: {dynamics_type}")