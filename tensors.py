import torch

PRECISION = torch.finfo(torch.float32).eps


def check_mat_diag(mat: torch.Tensor) -> bool:
    """
    Check if all elements of a batch of square matrices are diagonal
    """
    assert mat.shape[-1] == mat.shape[-2], "Input tensor must be a batch of square matrices"

    return ((mat - torch.diag_embed(mat.diagonal(dim1=-1, dim2=-2), dim1=-1, dim2=-2)).abs() < PRECISION).all()