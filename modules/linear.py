from typing import Optional
import torch
import bound_propagation as bp

__all__ = ['BoundLinear', 'Linear']

class Linear(torch.nn.Linear):
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, **kwargs):
        super(Linear, self).__init__(weight.size(-2), weight.size(-1), bias=bias is not None)
        with torch.no_grad():
            self.weight.copy_(weight)
            if bias is not None:
                self.bias.copy_(bias)

class BoundLinear(bp.BoundLinear):
    def __init__(self, *args, **kwargs):
        super(BoundLinear, self).__init__(*args, **kwargs)

    @bp.activation.assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        center, diff = bounds.center, bounds.width / 2

        if torch.logical_and(bounds.lower.isneginf(), ~bounds.upper.isinf()).any():
            upper = bounds.upper.matmul(self.module.weight)
            if self.module.bias is not None:
                upper = upper + self.module.bias.unsqueeze(-2)

            w_diff = diff.matmul(self.module.weight.abs())

            lower = upper - w_diff
        elif torch.logical_and(~bounds.lower.isneginf(), bounds.upper.isinf()).any():
            lower = bounds.lower.matmul(self.module.weight)
            if self.module.bias is not None:
                lower + self.module.bias.unsqueeze(-2)

            w_diff = diff.matmul(self.module.weight.abs())

            upper = lower + w_diff
        else:
            lower, upper = bp.linear.ibp_forward_linear_jit(self.module.weight, self.module.bias, center, diff)

        return bp.IntervalBounds(bounds.region, lower, upper)