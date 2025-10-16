import torch
import bound_propagation as bp
from functools import wraps
from .utils import NotLinearizable

__all__ = ['BoundSigmoid', 'BoundIdentity', 'BoxedIndicator', 'BoundBoxedIndicator', 'BoundTanh']

def assert_bound_order(func, position=0, keyword='preactivation'):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if len(args) > position:
            bounds = args[position]
        else:
            bounds = kwargs[keyword]

        if torch.isnan(bounds.lower).any() or torch.isnan(bounds.upper).any() or \
            not torch.all(bounds.lower <= bounds.upper + 1e-6):
            raise ValueError(f"Bounds are not ordered: {bounds.lower} <= {bounds.upper}")

        return func(self, *args, **kwargs)

    return wrapper

class SigmoidTangentBisectionStrategy:
    def upper_tangent(self, bound_module, lower, upper):
        lower_act = bound_module(lower)

        def f_upper(d: torch.Tensor) -> torch.Tensor:
            a_slope = (bound_module(d) - lower_act) / (d - lower)
            a_derivative = bound_module.derivative(d)
            return a_slope - a_derivative

        # Bisection will return left and right bounds for d s.t. f_upper(d) is zero
        # Derivative of left bound will over-approximate the slope - hence a true bound
        d_upper, _ = bp.activation.bisection(torch.zeros_like(upper), upper, f_upper, num_iter=1000)
        return d_upper

    def lower_tangent(self, bound_module, lower, upper):
        upper_act = bound_module(upper)

        def f_lower(d: torch.Tensor) -> torch.Tensor:
            a_slope = (upper_act - bound_module(d)) / (upper - d)
            a_derivative = bound_module.derivative(d)
            return a_derivative - a_slope

        # Bisection will return left and right bounds for d s.t. f_lower(d) is zero
        # Derivative of right bound will over-approximate the slope - hence a true bound
        _, d_lower = bp.activation.bisection(lower, torch.zeros_like(lower), f_lower, num_iter=1000)
        return d_lower

class BoundSigmoid(bp.BoundSigmoid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tangent_strategy = SigmoidTangentBisectionStrategy()

    @assert_bound_order
    def strict_ibp_forward(self, bounds, intersection, save_relaxation=False, save_input_bounds=False):
        if save_relaxation:
            self.strict_alpha_beta(preactivation=bounds, intersection=intersection)
            self.bounded = True

        if save_input_bounds:
            self.input_bounds = bounds

        bounds = bp.IntervalBounds(bounds.region, self.module(bounds.lower), self.module(bounds.upper))
        intersection = self.module(intersection)

        return bounds, intersection

    @assert_bound_order
    def strict_alpha_beta(self, preactivation, intersection):
        lower, upper = preactivation.lower, preactivation.upper

        at_lower = torch.isclose(intersection, lower, atol=1e-5)
        at_upper = torch.isclose(intersection, upper, atol=1e-5)
        if not torch.logical_or(at_lower, at_upper).all():
            raise NotLinearizable

        zero_width, n, p, np = bp.activation.regimes(lower, upper)

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        # Use upper and lower in the bias to account for a small numerical difference between lower and upper
        # which ought to be negligible, but may still be present due to torch.isclose.
        self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, self(lower[zero_width])
        self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, self(upper[zero_width])

        lower_prime, upper_prime = self.derivative(lower), self.derivative(upper)

        inter_act = self.module(intersection)
        inter_prime = self.derivative(intersection)

        slope = (self(upper) - self(lower)) / (upper - lower)

        def add_linear(alpha, beta, mask, a, x, y, a_mask=True):
            if a_mask:
                a = a[mask]

            alpha[mask] = a
            beta[mask] = y[mask] - a * x[mask]

        ###################
        # Negative regime #
        ###################
        # Upper bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_upper, self.beta_upper, mask=n, a=slope, x=intersection, y=inter_act)

        # Lower bound
        # - Slope is sigma'(intersection) and it has to cross through sigma(intersection)
        add_linear(self.alpha_lower, self.beta_lower, mask=n, a=inter_prime, x=intersection, y=inter_act)

        ###################
        # Positive regime #
        ###################
        # Lower bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_lower, self.beta_lower, mask=p, a=slope, x=intersection, y=inter_act)

        # Upper bound
        # - Slope is sigma'(intersection) and it has to cross through sigma(intersection)
        add_linear(self.alpha_upper, self.beta_upper, mask=p, a=inter_prime, x=intersection, y=inter_act)

        #################
        # Crossing zero #
        #################
        # Upper bound #
        # If tangent to upper is below lower, then take direct slope between lower and upper
        direct_slope = np & ((slope < upper_prime))
        add_linear(self.alpha_upper, self.beta_upper, mask=direct_slope, a=slope, x=intersection, y=inter_act)

        # Elif intersection is in positive half-space, take tangent to intersection (i.e. upper)
        direct_tangent = np & ~(slope < upper_prime) & (intersection >= 0)
        add_linear(self.alpha_upper, self.beta_upper, mask=direct_tangent, a=inter_prime, x=intersection, y=inter_act)

        # Else use bisection to find upper bound on slope.
        implicit = np & ~(slope < upper_prime) & (intersection < 0)

        if torch.any(implicit):
            d = self.tangent_strategy.upper_tangent(self, lower.clamp(-1000, 1000)[implicit], upper.clamp(-1000, 1000)[implicit])

            # Slope has to attach to (intersection, sigma(intersection))
            add_linear(self.alpha_upper, self.beta_upper, mask=implicit, a=self.derivative(d), x=intersection, y=inter_act, a_mask=False)

        # Lower bound #
        # If tangent to lower is above upper, then take direct slope between lower and upper
        direct_slope = np & (slope < lower_prime)
        add_linear(self.alpha_lower, self.beta_lower, mask=direct_slope, a=slope, x=intersection, y=inter_act)

        # Elif intersection is in negative half-space, take tangent to intersection (i.e. lower)
        direct_tangent = np & ~(slope < lower_prime) & (intersection <= 0)
        add_linear(self.alpha_lower, self.beta_lower, mask=direct_tangent, a=inter_prime, x=intersection, y=inter_act)

        # Else take tangent
        implicit = np & ~(slope < lower_prime) & (intersection > 0)

        if torch.any(implicit):
            d = self.tangent_strategy.lower_tangent(self, lower.clamp(-1000, 1000)[implicit], upper.clamp(-1000, 1000)[implicit])

            # Slope has to attach to (intersection, sigma(intersection))
            add_linear(self.alpha_lower, self.beta_lower, mask=implicit, a=self.derivative(d), x=intersection, y=inter_act, a_mask=False)


class BoundTanh(BoundSigmoid):
    def derivative(self, x):
        return 1 - torch.tanh(x) ** 2


class BoundIdentity(bp.activation.BoundIdentity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def strict_ibp_forward(self, bounds, intersection, save_relaxation=False, save_input_bounds=False):
        bounds = self.ibp_forward(bounds, save_relaxation, save_input_bounds)
        intersection = self.module(intersection)
        return bounds, intersection


class BoxedIndicator(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()

        self.min = min
        self.max = max

    def forward(self, x, all=True):
        if all:
            mask = ((x >= self.min) & (x <= self.max)).all(dim=-1).unsqueeze(-1)
        else:
            mask = (x >= self.min) & (x <= self.max)
        return x * mask


class BoundBoxedIndicator(bp.BoundActivation):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

    def clear_relaxation(self):
        super().clear_relaxation()

    @assert_bound_order
    def strict_ibp_forward(self, bounds, intersection, save_relaxation=False, save_input_bounds=False):
        if save_relaxation:
            self.strict_alpha_beta(preactivation=bounds, intersection=intersection)
            self.bounded = True

        if save_input_bounds:
            self.input_bounds = bounds

        act_lower, act_upper =  self.module(bounds.lower, all=False),self.module(bounds.upper, all=False)
        mask = act_lower >= act_upper
        act_lower[mask] = 0.
        act_upper[mask] = 0.

        bounds = bp.IntervalBounds(bounds.region, act_lower, act_upper)

        intersection = self.module(intersection, all=False)

        return bounds, intersection

    def parameterize_alpha_beta(self, alpha_lower, alpha_upper, beta_lower, beta_upper):
        raise NotImplementedError

    def alpha_beta(self, preactivation):
        raise NotImplementedError

    @assert_bound_order
    def strict_alpha_beta(self, preactivation, intersection):
        """
        Adaptive is similar to :BoundReLU: with the adaptivity being applied to both bends

        :param self:
        :param preactivation:
        """
        lower, upper = preactivation.lower, preactivation.upper

        at_lower = torch.isclose(intersection, lower, atol=1e-5)
        at_upper = torch.isclose(intersection, upper, atol=1e-5)
        if not torch.logical_or(at_lower, at_upper).all():
            raise NotLinearizable

        zero_width, flat_lower, flat_upper, slope, lower_bend, upper_bend, full_range = bp.saturation.regimes(
            lower, upper, self.module.min, self.module.max)

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        act_lower, act_upper = self.module(lower, all=False), self.module(upper, all=False)

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
        self.alpha_lower[flat_lower] = 0
        self.alpha_upper[flat_lower] = 0

        # Flat upper
        self.alpha_lower[flat_upper] = 0
        self.alpha_upper[flat_upper] = 0

        # Slope
        self.alpha_lower[slope] = 1
        self.alpha_upper[slope] = 1

        z = (self.module(upper, all=False) - self.module(lower, all=False)) / (upper - lower)

        # Lower bend
        if min is not None:
            # intersect at lower (left corner)
            lower_bend_left = lower_bend & at_lower

            lower_bend_left_min = min[lower_bend_left] if torch.is_tensor(min) else torch.as_tensor(min)
            self.alpha_lower[lower_bend_left] = lower_bend_left_min.clip(max=0.) / (lower_bend_left_min - lower[lower_bend_left])
            self.alpha_upper[lower_bend_left] = lower_bend_left_min.clip(min=0.) / (lower_bend_left_min - lower[lower_bend_left])

            # intersect at upper (right corner)
            lower_bend_right = lower_bend & at_upper

            lower_bend_right_min = min[lower_bend_right] if torch.is_tensor(min) else torch.as_tensor(min)
            self.alpha_lower[lower_bend_right] = (act_upper[lower_bend_right] - lower_bend_right_min.clip(max=0.)) / (upper[lower_bend_right] - lower_bend_right_min)
            self.alpha_upper[lower_bend_right] = upper[lower_bend_right] / (upper[lower_bend_right] - lower_bend_right_min)

            # Correct for special cases:
            lower_bend_neg_pos = lower_bend & (lower <= 0.) & (upper >= 0.)
            self.alpha_upper[lower_bend_neg_pos] = z[lower_bend_neg_pos]

            lower_bend_right_pos_pos = lower_bend_right & (lower >= 0.) & (upper >= 0.)
            self.alpha_upper[lower_bend_right_pos_pos] = 1.

        # Upper bend
        if max is not None:
            # intersect at lower
            upper_bend_left = upper_bend & at_lower

            upper_bend_left_max = max[upper_bend_left] if torch.is_tensor(max) else torch.as_tensor(max)
            self.alpha_lower[upper_bend_left] = (upper_bend_left_max.clip(max=0.) - lower[upper_bend_left]) / (upper_bend_left_max - lower[upper_bend_left])
            self.alpha_upper[upper_bend_left] = (upper_bend_left_max.clip(min=0.) - act_lower[upper_bend_left]) / (upper_bend_left_max - lower[upper_bend_left])

            # intersect at upper
            upper_bend_right = upper_bend & at_upper

            upper_bend_right_max = max[upper_bend_right] if torch.is_tensor(max) else torch.as_tensor(max)
            self.alpha_lower[upper_bend_right] = - upper_bend_right_max.clip(max=0.) / (upper[upper_bend_right] - upper_bend_right_max)
            self.alpha_upper[upper_bend_right] = - upper_bend_right_max.clip(min=0) / (upper[upper_bend_right] - upper_bend_right_max)

            # correct for special cases
            upper_bend_pos_neg = upper_bend & (lower <= 0.) & (upper >= 0.)
            self.alpha_lower[upper_bend_pos_neg] = z[upper_bend_pos_neg]

        # Full range
        if self.module.min is not None and self.module.max is not None:
            # intersect at lower
            full_range_left = full_range & at_lower

            full_range_left_min = min[full_range_left] if torch.is_tensor(min) else torch.as_tensor(min)
            full_range_left_max = max[full_range_left] if torch.is_tensor(max) else torch.as_tensor(max)

            self.alpha_lower[full_range_left] = full_range_left_min.clip(max=0.) / (full_range_left_min - lower[full_range_left])
            self.alpha_upper[full_range_left] = full_range_left_max.clip(min=0.) / (full_range_left_max - lower[full_range_left])

            # correction
            full_range_left_pos = full_range_left & (lower > 0.)
            full_range_left_pos_min = min[full_range_left_pos] if torch.is_tensor(min) else torch.as_tensor(min)
            self.alpha_upper[full_range_left_pos] = full_range_left_pos_min / (full_range_left_pos_min - lower[full_range_left_pos])

            # intersect at upper
            full_range_right = full_range & at_upper

            full_range_right_min = min[full_range_right] if torch.is_tensor(min) else torch.as_tensor(min)
            full_range_right_max = max[full_range_right] if torch.is_tensor(max) else torch.as_tensor(max)

            self.alpha_lower[full_range_right] = -full_range_right_min.clip(max=0) / (upper[full_range_right] - full_range_right_min)
            self.alpha_upper[full_range_right] = -full_range_right_max.clip(min=0) / (upper[full_range_right] - full_range_right_max)

            # correction
            full_range_right_neg = full_range_right & (upper <= 0.)
            full_range_right_neg_max = max[full_range_right_neg] if torch.is_tensor(max) else torch.as_tensor(max)
            self.alpha_lower[full_range_right_neg] = -full_range_right_neg_max / (upper[full_range_right_neg] - full_range_right_neg_max)

        self.beta_lower[at_lower] = act_lower[at_lower] - lower[at_lower] * self.alpha_lower[at_lower]
        self.beta_upper[at_lower] = act_lower[at_lower] - lower[at_lower] * self.alpha_upper[at_lower]

        self.beta_lower[at_upper] = act_upper[at_upper] - upper[at_upper] * self.alpha_lower[at_upper]
        self.beta_upper[at_upper] = act_upper[at_upper] - upper[at_upper] * self.alpha_upper[at_upper]