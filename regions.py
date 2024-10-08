import torch
from bound_propagation import HyperRectangle

def generate_voronoi_partition(signatures):

    print("CURRENTLY ONLY WORKS FOR 1D")
    delta = signatures.locs.squeeze()[1:] - signatures.locs.squeeze()[:-1]
    regions_lower = torch.cat([torch.tensor([-float('inf')]), signatures.locs.squeeze()[1:] - delta / 2])
    regions_upper = torch.cat([signatures.locs.squeeze()[:-1] + delta / 2, torch.tensor([float('inf')])])

    regions = HyperRectangle(regions_lower.unsqueeze(-1), regions_upper.unsqueeze(-1))

    return regions