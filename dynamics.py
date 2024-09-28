import math
import torch
from abc import ABC, abstractmethod

class _Dynamics(ABC):

    @abstractmethod
    def interval_approximation(self, region: torch.Tensor):
        pass

    def __call__(self, *args, **kwargs):
        pass


class GaussianDynamics(_Dynamics):
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x: torch.Tensor):
        coefficient = 1 / (self.sigma * math.sqrt(2 * math.pi))
        exponent = (-0.5 * ((x - self.mu) / self.sigma).pow(2)).exp()
        return coefficient * exponent

    def interval_approximation(self, region: torch.Tensor):

        zero_tensor = torch.tensor([0])

        points_1 = region.clone()
        points_2 = region.clone()

        if (region[0] <= 0) & (region[1] >= 0):
            points_1 = torch.cat((region, zero_tensor))
        if (region[0] <= 0) & (region[1] >= 0):
            points_2 = torch.cat((region, zero_tensor))

        # Evaluate the dynamics at all boundary points
        values_1 = self(points_1)
        values_2 = self(points_2)

        # Compute the maximum absolute difference between all pairs
        max_value = (torch.max(torch.abs(values_1.unsqueeze(0) - values_2.unsqueeze(1))))
        return max_value.item()

    @staticmethod
    def global_lipschitz():
        return math.exp(-1 / 2) / 2 * math.pi


class LinearAdversarial2DDynamics(_Dynamics):
    def __init__(self, A: torch.Tensor):
        self.A = A

    def __call__(self, x: torch.Tensor):
        return torch.matmul(x, self.A.T)

    def global_lipschitz(self):
        diagonal = torch.diag(self.A)  #Assuming diagonal A for now
        return torch.max(torch.abs(diagonal))