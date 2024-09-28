import torch

def compute_bound(lambd, signature_probs, beta, projection_matrix, budget):
    # \TODO optimize memory w.r.t. projection_matrix
    constant_term = beta
    value_matrix = constant_term - lambd * projection_matrix[:, None]

    # Take the max over the computed value_matrix
    max_values = value_matrix.max(0).values

    # Compute the outer_sum using vectorized operations
    outer_sum = torch.sum(signature_probs * max_values)
    outer_sum += lambd * budget

    return torch.sqrt(outer_sum)