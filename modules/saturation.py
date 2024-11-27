import torch
import bound_propagation as bp

__all__ = ['BoundClamp']

class BoundClamp(bp.BoundClamp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    @bp.activation.assert_bound_order
    def alpha_beta(self, preactivation):
        """
        Adaptive is similar to :BoundReLU: with the adaptivity being applied to both bends

        :param self:
        :param preactivation:
        """
        lower, upper = preactivation.lower, preactivation.upper
        zero_width, flat_lower, flat_upper, slope, lower_bend, upper_bend, full_range = bp.saturation.regimes(
            lower, upper, self.module.min, self.module.max)

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        act_lower, act_upper = self(lower), self(upper)

        # Use upper and lower in the bias to account for a small numerical difference between lower and upper
        # which ought to be negligible, but may still be present due to torch.isclose.
        if zero_width.any():
            self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, act_lower[zero_width]
            self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, act_upper[zero_width]

        min, max = self.module.min, self.module.max
        if min is not None and torch.is_tensor(min):
            min = min.view(*[1 for _ in range(self.alpha_lower.dim() - 1)], -1).expand_as(self.alpha_lower)

        if max is not None and torch.is_tensor(max):
            max = max.view(*[1 for _ in range(self.alpha_lower.dim() - 1)], -1).expand_as(self.alpha_lower)

        # Flat lower
        flat_lower_min = min[flat_lower] if min is not None and torch.is_tensor(min) else min
        self.alpha_lower[flat_lower], self.beta_lower[flat_lower] = 0, 0 if min is None else flat_lower_min
        self.alpha_upper[flat_lower], self.beta_upper[flat_lower] = 0, 0 if min is None else flat_lower_min

        # Flat upper
        flat_upper_max = max[flat_upper] if max is not None and torch.is_tensor(max) else max
        self.alpha_lower[flat_upper], self.beta_lower[flat_upper] = 0, 0 if max is None else flat_upper_max
        self.alpha_upper[flat_upper], self.beta_upper[flat_upper] = 0, 0 if max is None else flat_upper_max

        # Slope
        self.alpha_lower[slope], self.beta_lower[slope] = 1, 0
        self.alpha_upper[slope], self.beta_upper[slope] = 1, 0

        z = (self(upper) - self(lower)) / (upper - lower)

        # Lower bend
        if min is not None:
            self.alpha_lower[lower_bend] = 1.
            self.beta_lower[lower_bend] = act_upper[lower_bend] - upper[lower_bend] * 1.

            self.alpha_upper[lower_bend] = z[lower_bend]
            self.beta_upper[lower_bend] = act_upper[lower_bend] - upper[lower_bend] * z[lower_bend]

            self.unstable_lower = lower_bend
            self.unstable_slope_lower = z[lower_bend].detach().clone().requires_grad_()

        # Upper bend
        if max is not None:
            self.alpha_lower[upper_bend] = z[upper_bend]
            self.beta_lower[upper_bend] = act_lower[upper_bend] - lower[upper_bend] * z[upper_bend]

            self.alpha_upper[upper_bend] = 1.
            self.beta_upper[upper_bend] = act_lower[upper_bend] - lower[upper_bend] * 1.

            self.unstable_upper = upper_bend
            self.unstable_slope_upper = z[upper_bend].detach().clone().requires_grad_()

        # Full range
        if self.module.min is not None and self.module.max is not None:
            full_range_min = min[full_range] if torch.is_tensor(min) else min
            full_range_max = max[full_range] if torch.is_tensor(max) else max

            act_full_range_min = self(full_range) if torch.is_tensor(min) else self(torch.as_tensor(full_range_min))
            act_full_range_max = self(full_range) if torch.is_tensor(max) else self(torch.as_tensor(full_range_max))

            z_lower = (full_range_max - full_range_min) / (upper[full_range] - full_range_min)
            self.alpha_lower[full_range] = z_lower
            self.beta_lower[full_range] = act_full_range_min - full_range_min * z_lower

            z_upper = (full_range_max - full_range_min) / (full_range_max - lower[full_range])
            self.alpha_upper[full_range] = z_upper
            self.beta_upper[full_range] = act_full_range_max - full_range_max * z_upper
