import math
import torch
from abc import ABC, abstractmethod
from typing import Union

from regions import HyperRectangularVoronoiPartition


class Dynamics(ABC):
    num_dims = None

    @abstractmethod
    def bound_lp2_norm_difference(self, voronoi_partition: HyperRectangularVoronoiPartition):
        """
        find matrix B such that ||f(x) - f(c_i)|| leq b_{ik} for all x in region R_k and c_i the representative
         point of R_i, with R_k and R_i the k-th and i-th element of voronoi_partition, respectively.

        CONVENTION: indexing regions over columns and points over rows

        :param voronoi_partition:
        :return: shape: (voronoi_partition.num_points, voronoi_partition.num_points)
        """
        pass

    def __call__(self, x: torch.Tensor):
        """
        Function Evaluation
        :param x:
        :return:
        """
        return x

    @property
    def global_lipschitz(self):
        """
        Global Lipschitz constant
        :return:
        """
        return None

class _LogConcaveDynamics(Dynamics):
    def __init__(self):
        super().__init__()

    def bound_lp2_norm_difference(self, voronoi_partition: HyperRectangularVoronoiPartition):
        if self.num_dims > 1:
            raise NotImplementedError # Requires checking all vertices, not only lower and upper

        extremum = voronoi_partition.points.clone()
        mask_extremum = torch.logical_and(voronoi_partition.lower <= self.location_extremum,
                                          self.location_extremum >= voronoi_partition.upper)
        extremum[mask_extremum] = self.extremum

        return torch.max(torch.max(
            torch.norm(self(voronoi_partition.lower).unsqueeze(0) - self(voronoi_partition.points).unsqueeze(1), p=2, dim=-1),
            torch.norm(self(voronoi_partition.upper).unsqueeze(0) - self(voronoi_partition.points).unsqueeze(1), p=2, dim=-1)),
            torch.norm(self(extremum).unsqueeze(0) - self(voronoi_partition.points).unsqueeze(1), p=2, dim=-1)
        )

    @property
    def location_extremum(self):
        return torch.ones(self.num_dims).fill_(torch.nan)

    @property
    def extremum(self):
        return self(self.location_extremum)


class _MonotoneDynamics(Dynamics):
    def __init__(self):
        super().__init__()

    def bound_lp2_norm_difference(self, voronoi_partition: HyperRectangularVoronoiPartition):
        return torch.max(
            torch.norm(self(voronoi_partition.lower).unsqueeze(0) - self(voronoi_partition.points).unsqueeze(1), p=2, dim=-1),
            torch.norm(self(voronoi_partition.upper).unsqueeze(0) - self(voronoi_partition.points).unsqueeze(1), p=2, dim=-1)
        )

class GaussianDynamics1d(_LogConcaveDynamics):
    def __init__(self, loc: float, scale: float, **kwargs):
        self.num_dims = 1
        self.loc = torch.tensor(loc)
        self.scale = torch.tensor(scale)
        self.gaussian_distribution = torch.distributions.Normal(loc=loc, scale=scale)
        super(GaussianDynamics1d, self).__init__()

    def __call__(self, x: torch.Tensor):
        log_pdf = self.gaussian_distribution.log_prob(x)
        return torch.exp(log_pdf)

    @property
    def location_max_value(self):
        return self.loc

    @property
    def global_lipschitz(self):
        if not (self.scale <= 1).all():
            raise NotImplementedError
        else:
            return math.exp(-1 / 2) / math.sqrt(2 * math.pi)

class ChaoticDynamics(_LogConcaveDynamics):
    num_dims = 1
    def __init__(self, r: float, **kwargs):
        self.r = r
        super(ChaoticDynamics, self).__init__()

    def __call__(self, x: torch.Tensor):
        return torch.where((x > 0) & (x < 1), self.r * x * (1 - x), torch.zeros_like(x))

    @property
    def global_lipschitz(self):
        return self.r

    @property
    def location_max_value(self):
        return torch.tensor(1 / (2 * self.r))

class LinearDynamics(_MonotoneDynamics):
    def __init__(self, diagonal: Union[torch.Tensor, list], **kwargs):
        if isinstance(diagonal, list):
            diagonal = torch.tensor(diagonal)

        self.num_dims = diagonal.size(0)
        self.mat = torch.diag(diagonal)
        self.mat_is_diagonal = True
        super(LinearDynamics, self).__init__()

    def __call__(self, x: torch.Tensor):
        y = torch.matmul(x.clip(-2., 2.), self.mat.T)
        return y

    @property
    def global_lipschitz(self):
        if self.mat_is_diagonal:
            return self.mat.diagonal().abs().max()
        else:
            raise NotImplementedError

def get_dynamics(dynamics_type: str, **kwargs):
    if dynamics_type == 'GaussianDynamics1d':
        return GaussianDynamics1d(**kwargs)
    elif dynamics_type == 'ChaoticDynamics':
        return ChaoticDynamics(**kwargs)
    elif dynamics_type == 'LinearDynamics':
        return LinearDynamics(**kwargs)
    else:
        raise ValueError(f"Unknown dynamics: {dynamics_type}")