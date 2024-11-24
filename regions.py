import torch

class HyperRectangularVoronoiPartition:
    def __init__(self, locs: torch.Tensor):
        if not self._are_locs_in_grid(locs):
            raise ValueError("locs must be in a grid")
        self._locs = locs
        self._num_dims = locs.size(-1)
        self._lower = self._get_lower()
        self._upper = self._get_upper()

        assert (self._lower <= self._upper).all()


    @property
    def num_locs(self):
        return self.locs.size(0)

    @property
    def num_dims(self):
        return self._num_dims

    @property
    def locs(self):
        return self._locs

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @staticmethod
    def _are_locs_in_grid(locs: torch.Tensor):
        #unique_elements_per_n = torch.tensor([torch.unique(locs[:, i]).numel() for i in range(locs.size(1))])
        #return unique_elements_per_n.prod() == torch.unique(locs,dim=0).size(0)

        #TODO: @Steven, is this check above correct?
        #TODO: When d=2, we have a GMM with 81 locs: 12 * 11 vs 81
        return True

    def _get_upper(self):
        pos_diff = (self._locs.unsqueeze(-3) - self._locs.unsqueeze(-2)).clip(0, torch.inf)
        mask = pos_diff == 0.
        pos_diff[mask] = torch.inf

        upper = self._locs + 0.5 * pos_diff.min(dim=-2).values

        return upper

    def _get_lower(self):
        neg_diff = (self._locs.unsqueeze(-3) - self._locs.unsqueeze(-2)).clip(-torch.inf, 0)
        mask = neg_diff == 0.
        neg_diff[mask] = -torch.inf

        lower = self._locs + 0.5 * neg_diff.max(dim=-2).values

        return lower

    @property
    def center(self):
        return (self.upper + self.lower) / 2

    @property
    def width(self):
        return self.upper - self.lower
