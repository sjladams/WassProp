import torch
import bound_propagation as bp
from .utils import is_vertice

__all__ = ['BoundClamp']

class BoundClamp(bp.BoundClamp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @bp.activation.assert_bound_order
    def strict_ibp_forward(self, bounds, intersection, save_relaxation=False, save_input_bounds=False):
        if save_relaxation:
            self.strict_alpha_beta(preactivation=bounds, intersection=intersection)
            self.bounded = True

        if save_input_bounds:
            self.input_bounds = bounds

        bounds = bp.IntervalBounds(bounds.region, self.module(bounds.lower), self.module(bounds.upper))

        intersection = self(intersection)
        if not is_vertice(bounds, intersection):
            raise NotImplementedError

        return bounds, intersection


    @bp.activation.assert_bound_order
    def strict_alpha_beta(self, preactivation, intersection):
        """
        Adaptive is similar to :BoundReLU: with the adaptivity being applied to both bends

        :param self:
        :param preactivation:
        """
        lower, upper = preactivation.lower, preactivation.upper

        at_lower = torch.isclose(intersection, lower)
        at_upper = torch.isclose(intersection, upper)
        assert torch.logical_or(at_lower, at_upper).all()

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
            self.alpha_upper[lower_bend] = z[lower_bend]
            self.beta_upper[lower_bend] = act_lower[lower_bend] - lower[lower_bend] * z[lower_bend]

            # intersect at lower
            lower_bend_left = lower_bend & at_lower
            self.alpha_lower[lower_bend_left] = 0.
            self.beta_lower[lower_bend_left] = act_lower[lower_bend_left]

            # intersect at upper
            lower_bend_right = lower_bend & at_upper
            self.alpha_lower[lower_bend_right] = 1.
            self.beta_lower[lower_bend_right] = act_upper[lower_bend_right] - upper[lower_bend_right] * 1.

        # Upper bend
        if max is not None:
            self.alpha_lower[upper_bend] = z[upper_bend]
            self.beta_lower[upper_bend] = act_lower[upper_bend] - lower[upper_bend] * z[upper_bend]

            # intersect at lower
            upper_bend_left = upper_bend & at_lower
            self.alpha_upper[upper_bend_left] = 1.
            self.beta_upper[upper_bend_left] = act_lower[upper_bend_left] - lower[upper_bend_left] * 1.

            # intersect at upper
            upper_bend_right = upper_bend & at_upper
            self.alpha_upper[upper_bend_right] = 0.
            self.beta_upper[upper_bend_right] = act_upper[upper_bend_right]

        # Full range
        if self.module.min is not None and self.module.max is not None:
            # intersect at lower
            full_range_left = full_range & at_lower

            full_range_left_min = min[full_range_left] if torch.is_tensor(min) else torch.as_tensor(min)
            full_range_left_max = max[full_range_left] if torch.is_tensor(max) else torch.as_tensor(max)

            z_full_range_left = (self(full_range_left_max) - self(full_range_left_min)) / (full_range_left_max - lower[full_range_left])

            self.alpha_lower[full_range_left] = 0.
            self.beta_lower[full_range_left] = act_lower[full_range_left]

            self.alpha_upper[full_range_left] = z_full_range_left
            self.beta_upper[full_range_left] = act_lower[full_range_left] - lower[full_range_left] * z_full_range_left

            # intersect at upper
            full_range_right = full_range & at_upper

            full_range_right_min = min[full_range_right] if torch.is_tensor(min) else torch.as_tensor(min)
            full_range_right_max = max[full_range_right] if torch.is_tensor(max) else torch.as_tensor(max)

            z_full_range_right = (self(full_range_right_max) - self(full_range_right_min)) / (upper[full_range_right] - full_range_right_min)

            self.alpha_lower[full_range_right] = z_full_range_right
            self.beta_lower[full_range_right] = act_upper[full_range_right] - upper[full_range_right] * z_full_range_right

            self.alpha_upper[full_range_right] = 0.
            self.beta_upper[full_range_right] = act_upper[full_range_right]


