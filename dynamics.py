import torch
from typing import Union
import bound_propagation as bp

from torch_modules import ScalarMult, ScalarAdd


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
    def __init__(self, r: float, **kwargs):
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
    def __init__(self, mat: Union[torch.Tensor, list], **kwargs):
        if isinstance(mat, list):
            mat = torch.tensor(mat)

        self.num_dims = mat.size(0)
        self._mat = mat

        linear = torch.nn.Linear(self.num_dims, self.num_dims, bias=False)
        with torch.no_grad():
            linear.weight.copy_(mat)

        super(LinearDynamics, self).__init__(linear)

    @property
    def global_lipschitz(self):
        return torch.linalg.svd(self._mat).S[0]

class LinearDiagonalDynamics(LinearDynamics):
    def __init__(self, diagonal: Union[torch.Tensor, list], **kwargs):
        if isinstance(diagonal, list):
            diagonal = torch.tensor(diagonal)

        super(LinearDiagonalDynamics, self).__init__(torch.diag(diagonal))

class BoundedLinearDynamics(Dynamics):
    def __init__(self,
                 mat: Union[torch.Tensor, list],
                 lower_bound: Union[float, torch.Tensor, list],
                 upper_bound: Union[float, torch.Tensor, list],
                 **kwargs):
        linear_dynamics = LinearDynamics(mat)

        super(BoundedLinearDynamics, self).__init__(linear_dynamics, bp.Clamp(lower_bound, upper_bound))

class BoundedLinearDiagonalDynamics(BoundedLinearDynamics):
    def __init__(self, diagonal: Union[torch.Tensor, list],
                 lower_bound: Union[float, torch.Tensor, list],
                 upper_bound: Union[float, torch.Tensor, list],
                 **kwargs):
        if isinstance(diagonal, list):
            diagonal = torch.tensor(diagonal)

        super(BoundedLinearDiagonalDynamics, self).__init__(torch.diag(diagonal), lower_bound, upper_bound)


class SigmoidDynamics(Dynamics):
    def __init__(self, num_dims: int = 1, **kwargs):
        super(SigmoidDynamics, self).__init__(torch.nn.Sigmoid())
        self.num_dims = num_dims

    @property
    def global_lipschitz(self):
        return 0.25

class LinearSigmoidDynamics(Dynamics):
    def __init__(self, diagonal: Union[torch.Tensor, list], **kwargs):
        if isinstance(diagonal, list):
            diagonal = torch.tensor(diagonal)
        self.num_dims = diagonal.size(0)
        self._diagonal = diagonal

        super(LinearSigmoidDynamics, self).__init__(
            LinearDiagonalDynamics(diagonal, min=-torch.inf, max=torch.inf),
            SigmoidDynamics(self.num_dims)
        )

    @property
    def global_lipschitz(self):
        return self._diagonal.abs().max() * 0.25

def get_dynamics(dynamics_type: str, **kwargs):
    if dynamics_type == 'LogisticMap':
        return LogisticMap(**kwargs)
    elif dynamics_type == 'LinearDynamics':
        return LinearDynamics(**kwargs)
    elif dynamics_type == 'BoundedLinearDynamics':
        return BoundedLinearDynamics(**kwargs)
    elif dynamics_type == 'LinearDiagonalDynamics':
        return LinearDiagonalDynamics(**kwargs)
    elif dynamics_type == 'BoundedLinearDiagonalDynamics':
        return BoundedLinearDiagonalDynamics(**kwargs)
    elif dynamics_type == 'SigmoidDynamics':
        return SigmoidDynamics(**kwargs)
    elif dynamics_type == 'LinearSigmoidDynamics':
        return LinearSigmoidDynamics(**kwargs)
    else:
        raise ValueError(f"Unknown dynamics: {dynamics_type}")