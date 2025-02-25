from typing import Optional
import torch
import bound_propagation as bp

__all__ = ['BoundLinear', 'Linear', 'Identity']


def nan_matmul(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    elem_mul = torch.einsum('ij,...j->...ij', mat, vec)
    elem_mul = torch.nan_to_num(elem_mul, posinf=torch.inf, neginf=-torch.inf)
    return elem_mul.sum(-1)


class Linear(torch.nn.Linear):
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, **kwargs):
        super(Linear, self).__init__(weight.size(-1), weight.size(-2), bias=bias is not None)
        with torch.no_grad():
            self.weight.copy_(weight)
            if bias is not None:
                self.bias.copy_(bias)

class Identity(Linear):
    def __init__(self, in_features: int):
        super(Identity, self).__init__(torch.eye(in_features))


class BoundLinear(bp.BoundLinear):
    def __init__(self, *args, **kwargs):
        super(BoundLinear, self).__init__(*args, **kwargs)

    @bp.activation.assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        center, diff = bounds.center, bounds.width / 2

        if bounds.lower.isinf().any() or bounds.upper.isinf().any():
            upper = nan_matmul(self.module.weight, bounds.upper)
            lower = nan_matmul(self.module.weight, bounds.lower)

            if self.module.bias is not None:
                upper = upper + self.module.bias.unsqueeze(-2)
                lower = lower + self.module.bias.unsqueeze(-2)

            w_diff = nan_matmul(self.module.weight.abs(), diff)

            lower = torch.max(torch.nan_to_num(upper - w_diff, nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf), lower)
            upper = torch.min(torch.nan_to_num(lower + w_diff, nan=torch.inf, posinf=torch.inf, neginf=-torch.inf), upper)
        else:
            lower, upper = bp.linear.ibp_forward_linear_jit(self.module.weight, self.module.bias, center, diff)

        return bp.IntervalBounds(bounds.region, lower, upper)