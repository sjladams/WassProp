import torch
from abc import ABC, abstractmethod
from typing import Union
from bound_propagation import Sub, Mul, Clamp

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

        clamp_0_1 = Sub(torch.nn.ReLU(), torch.nn.Sequential(ScalarAdd(self.num_dims, -1), torch.nn.ReLU()))

        super(LogisticMap, self).__init__(
            Mul(clamp_0_1, torch.nn.Sequential(clamp_0_1, ScalarAdd(self.num_dims, -1))),
            ScalarMult(self.num_dims, -r)
        )

    @property
    def global_lipschitz(self):
        return self.r

class BoundedLinearDiagonalDynamics(Dynamics):
    def __init__(self, diagonal: Union[torch.Tensor, list], min=-2., max=2., **kwargs):
        if isinstance(diagonal, list):
            diagonal = torch.tensor(diagonal)
        self.num_dims = diagonal.size(0)
        self._diagonal = diagonal

        linear = torch.nn.Linear(self.num_dims, self.num_dims, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.diag(diagonal))

        super(BoundedLinearDiagonalDynamics, self).__init__(linear, Clamp(min, max))

    @property
    def global_lipschitz(self):
        return self._diagonal.abs().max()

def get_dynamics(dynamics_type: str, **kwargs):
    if dynamics_type == 'LogisticMap':
        return LogisticMap(**kwargs)
    elif dynamics_type == 'BoundedLinearDiagonalDynamics':
        return BoundedLinearDiagonalDynamics(**kwargs)
    else:
        raise ValueError(f"Unknown dynamics: {dynamics_type}")