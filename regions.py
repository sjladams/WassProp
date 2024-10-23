import torch


class HyperRectangularVoronoiPartition:
    def __init__(self, locs: torch.Tensor, loc_shell: torch.Tensor, shell: torch.Tensor):
        if not self._are_locs_in_grid(locs):
            raise ValueError("locs must be in a grid")
        self._loc_shell = loc_shell
        self._shell = shell
        self._locs = locs
        self._lower = self._get_lower()
        self._upper = self._get_upper()


    @property
    def num_locs(self):
        return self.locs.size(0)

    @property
    def locs(self):
        return torch.cat((self._locs, self._loc_shell.unsqueeze(-2)), dim=-2)

    # @locs.setter
    # def locs(self, locs: torch.Tensor):
    #     if not self._are_locs_in_grid(locs):
    #         raise ValueError("locs must be in a grid")
    #     self._locs = locs
    #     self._lower = self._get_lower()
    #     self._upper = self._get_upper()

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @staticmethod
    def _are_locs_in_grid(locs: torch.Tensor):
        unique_elements_per_n = torch.tensor([torch.unique(locs[:, i]).numel() for i in range(locs.size(1))])
        return unique_elements_per_n.prod() == torch.unique(locs,dim=0).size(0)

    def _get_upper(self):
        pos_diff = (self._locs.unsqueeze(-3) - self._locs.unsqueeze(-2)).clip(0, torch.inf)
        mask = pos_diff == 0.
        pos_diff[mask] = self._shell[..., 1].unsqueeze(0).expand(pos_diff.shape)[mask]
        return torch.cat((self._locs + 0.5 * pos_diff.min(dim=-2).values, self._shell[..., 1].unsqueeze(-2)), dim=-2)

    def _get_lower(self):
        neg_diff = (self._locs.unsqueeze(-3) - self._locs.unsqueeze(-2)).clip(-torch.inf, 0)
        mask = neg_diff == 0.
        neg_diff[mask] = self._shell[..., 0].unsqueeze(0).expand(neg_diff.shape)[mask]
        return torch.cat((self._locs + 0.5 * neg_diff.max(dim=-2).values, self._shell[..., 0].unsqueeze(-2)), dim=-2)

    @property
    def center(self):
        return (self.upper + self.lower) / 2

    @property
    def width(self):
        return self.upper - self.lower
