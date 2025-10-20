import torch


def is_mat_diag(mat: torch.Tensor) -> bool:
    """
    Check if all elements of a batch of matrices are diagonal
    """
    if mat.size(-1) != mat.size(-2):
        return False
    else:
        return torch.equal(torch.zeros_like(mat), mat - torch.diag_embed(mat.diagonal(dim1=-2,dim2=-1), dim1=-2, dim2=-1))
