import torch

PRECISION = torch.finfo(torch.float32).eps


def check_mat_diag(mat: torch.Tensor) -> bool:
    """
    Check if all elements of a batch of square matrices are diagonal
    """
    return not (mat - mat.diagonal() > PRECISION).any()