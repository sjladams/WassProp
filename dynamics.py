import math
import torch
from abc import ABC, abstractmethod
from bound_propagation import HyperRectangle


class Dynamics(ABC):

    @abstractmethod
    def interval_approximation(self, regions: HyperRectangle):
        pass

    def __call__(self, x: torch.Tensor):
        """
        Function Evaluation
        :param x:
        :return:
        """
        return x

class _MonotoneDynamics(Dynamics):
    def __init__(self):
        super(_MonotoneDynamics, self).__init__()

    def interval_approximation(self, regions: HyperRectangle):
        y_lower = (self(regions.lower) - self(regions.center)).abs()
        y_upper = (self(regions.upper) - self(regions.center)).abs()
        return torch.max(y_lower, y_upper)


class GaussianDynamics1d(_MonotoneDynamics):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.loc = loc
        self.scale = scale
        self.gaussian_distribution = torch.distributions.Normal(loc=loc, scale=scale)
        super(GaussianDynamics1d, self).__init__()

    def __call__(self, x: torch.Tensor):
        log_pdf = self.gaussian_distribution.log_prob(x)
        return torch.exp(log_pdf)

    def global_lipschitz(self):
        # \TODO
        if not (self.scale <= 1).all():
            raise NotImplementedError
        else:
            return math.exp(-1 / 2) / math.sqrt(2 * math.pi)

class ChaoticDynamics(_MonotoneDynamics):
    def __init__(self, r: float):
        self.r = r
        super(ChaoticDynamics, self).__init__()

    def __call__(self, x: torch.Tensor):
        return torch.where((x > 0) & (x < 1), self.r * x * (1 - x), torch.zeros_like(x))

    def global_lipschitz(self):
        return self.r


class LinearDynamics(_MonotoneDynamics):
    def __init__(self, mat: torch.Tensor):
        self.mat = mat
        self.mat_is_diagonal = not (mat - mat.diagonal() > 0).any()
        super(LinearDynamics, self).__init__()

    def __call__(self, x: torch.Tensor):
        return torch.matmul(x, self.mat.T)

    def global_lipschitz(self):
        if self.mat_is_diagonal:
            return self.mat.diagonal().abs().max().values()