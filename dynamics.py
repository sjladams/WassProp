import math
import torch
from abc import ABC, abstractmethod

from regions import HyperRectangularVoronoiPartition


class Dynamics(ABC):
    num_dims = None

    @abstractmethod
    def bound_lp2_norm_difference(self, voronoi_partition: HyperRectangularVoronoiPartition):
        """
        find b such that ||f(x) - f(c)|| leq b for all x in regions
        :param voronoi_partition:
        :return:
        """
        pass

    def __call__(self, x: torch.Tensor):
        """
        Function Evaluation
        :param x:
        :return:
        """
        return x

class _ConvexDynamics(Dynamics):
    def __init__(self):
        super(_ConvexDynamics, self).__init__()

    def bound_lp2_norm_difference(self, voronoi_partition: HyperRectangularVoronoiPartition):
        bound = torch.max(
            torch.norm(self(voronoi_partition.lower) - self(voronoi_partition.points), p=2, dim=-1),
            torch.norm(self(voronoi_partition.upper) - self(voronoi_partition.points), p=2, dim=-1)
        )
        fmax_in_region = torch.logical_and(voronoi_partition.lower <= self.location_max_value,
                                           self.location_max_value <= voronoi_partition.upper
                                           ).all(dim=-1)
        bound[fmax_in_region] = torch.norm(self.max_value - self(voronoi_partition.points[fmax_in_region]), p=2, dim=-1)
        return bound

    @property
    def location_max_value(self):
        return torch.ones(self.num_dims).fill_(torch.nan)

    @property
    def max_value(self):
        return self(self.location_max_value)


class GaussianDynamics1d(_ConvexDynamics):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.num_dims = loc.size(0)
        self.loc = loc
        self.scale = scale
        self.gaussian_distribution = torch.distributions.Normal(loc=loc, scale=scale)
        super(GaussianDynamics1d, self).__init__()

    def __call__(self, x: torch.Tensor):
        log_pdf = self.gaussian_distribution.log_prob(x)
        return torch.exp(log_pdf)

    @property
    def location_max_value(self):
        return self.loc

    def global_lipschitz(self):
        # \TODO
        if not (self.scale <= 1).all():
            raise NotImplementedError
        else:
            return math.exp(-1 / 2) / math.sqrt(2 * math.pi)

class ChaoticDynamics(_ConvexDynamics):
    num_dims = 1
    def __init__(self, r: float):
        self.r = r
        super(ChaoticDynamics, self).__init__()

    def __call__(self, x: torch.Tensor):
        return torch.where((x > 0) & (x < 1), self.r * x * (1 - x), torch.zeros_like(x)) # @Eduardo, why not simply take self.r * x * (1 - x) ??

    def global_lipschitz(self):
        return self.r

    @property
    def location_max_value(self):
        return torch.tensor(1 / (2 * self.r))

class LinearDynamics(_ConvexDynamics):
    def __init__(self, mat: torch.Tensor):
        self.num_dims = mat.size(0)
        self.mat = mat
        self.mat_is_diagonal = not (mat - mat.diagonal() > 0).any()
        super(LinearDynamics, self).__init__()

    def __call__(self, x: torch.Tensor):
        return torch.matmul(x, self.mat.T)

    def global_lipschitz(self):
        if self.mat_is_diagonal:
            return self.mat.diagonal().abs().max().values()