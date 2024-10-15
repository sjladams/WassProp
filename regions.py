import torch


class HyperRectangularVoronoiPartition:
    def __init__(self, points: torch.Tensor):
        if not self._are_points_in_grid(points):
            raise ValueError("Points must be in a grid")
        self._points = points
        self._lower = self._get_lower()
        self._upper = self._get_upper()

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points: torch.Tensor):
        if not self._are_points_in_grid(points):
            raise ValueError("Points must be in a grid")
        self._points = points
        self._lower = self._get_lower()
        self._upper = self._get_upper()

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @staticmethod
    def _are_points_in_grid(points: torch.Tensor):
        unique_elements_per_n = torch.tensor([torch.unique(points[:, i]).numel() for i in range(points.size(1))])
        return unique_elements_per_n.prod() == torch.unique(points,dim=0).size(0)

    def _get_upper(self):
        pos_diff = (self.points.unsqueeze(-3) - self.points.unsqueeze(-2)).clip(0, torch.inf)
        pos_diff[pos_diff == 0.] = torch.inf
        return self.points + 0.5 * pos_diff.min(dim=-2).values

    def _get_lower(self):
        neg_diff = (self.points.unsqueeze(-3) - self.points.unsqueeze(-2)).clip(-torch.inf, 0)
        neg_diff[neg_diff == 0.] = -torch.inf
        return self.points + 0.5 * neg_diff.max(dim=-2).values


    @property
    def center(self):
        return (self.upper + self.lower) / 2

    @property
    def width(self):
        return self.upper - self.lower
