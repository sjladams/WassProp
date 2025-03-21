import torch
import bound_propagation as bp
import math
from .utils import is_vertice

__all__ = ['BoundSin']


class SinTangentBisectionStrategy:
    @staticmethod
    def increasing_upper_tangent(bound_module, lower, bisection_lower, bisection_upper):
        lower_act = bound_module(lower)

        def f_lower(d: torch.Tensor) -> torch.Tensor:
            a_slope = (bound_module(d)- lower_act) / (d - lower)
            a_derivative = bound_module.derivative(d)
            return a_slope - a_derivative

        _, d_upper = bp.activation.bisection(bisection_lower, bisection_upper, f_lower, num_iter=100)
        return d_upper

    @staticmethod
    def increasing_lower_tangent(bound_module, upper, bisection_lower, bisection_upper):
        upper_act = bound_module(upper)

        def f_lower(d: torch.Tensor) -> torch.Tensor:
            a_slope = (upper_act - bound_module(d)) / (upper - d)
            a_derivative = bound_module.derivative(d)
            return a_derivative - a_slope

        # Bisection will return left and right bounds for d s.t. f_lower(d) is zero
        # Derivative of left bound will over-approximate the slope - hence a true bound
        d_lower, _ = bp.activation.bisection(bisection_lower, bisection_upper, f_lower, num_iter=100)
        return d_lower

    @staticmethod
    def decreasing_upper_tangent(bound_module, upper, bisection_lower, bisection_upper):
        upper_act = bound_module(upper)

        def f_upper(d: torch.Tensor) -> torch.Tensor:
            a_slope = (upper_act - bound_module(d)) / (upper - d)
            a_derivative = bound_module.derivative(d)
            return a_slope - a_derivative

        _, d_upper = bp.activation.bisection(bisection_lower, bisection_upper, f_upper, num_iter=100)
        return d_upper

    @staticmethod
    def decreasing_lower_tangent(bound_module, lower, bisection_lower, bisection_upper):
        lower_act = bound_module(lower)

        def f_upper(d: torch.Tensor) -> torch.Tensor:
            a_slope = (bound_module(d) - lower_act) / (d - lower)
            a_derivative = bound_module.derivative(d)
            return a_derivative - a_slope

        d_lower, _ = bp.activation.bisection(bisection_lower, bisection_upper, f_upper, num_iter=100)
        return d_lower


class BoundSin(bp.BoundSin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tangent_strategy = kwargs.get('sin_tangent_strategy', SinTangentBisectionStrategy())

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
        lower, upper = preactivation.lower, preactivation.upper

        at_lower = torch.isclose(intersection, lower)
        at_upper = torch.isclose(intersection, upper)
        assert torch.logical_or(at_lower, at_upper).all()

        zero_width, half_period, (_, increasing), (_, decreasing), crossing_peak, crossing_trough = \
            bp.activation.sine_like_regimes(lower, upper, period=self.period, zero_increasing=self.zero_increasing)

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        # Use upper and lower in the bias to account for a small numerical difference between lower and upper
        # which ought to be negligible, but may still be present due to torch.isclose.
        self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, self(lower[zero_width])
        self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, self(upper[zero_width])

        inter_act, inter_prime = self(intersection), self.derivative(intersection)

        def add_linear(alpha, beta, mask, a, x, y, a_mask=True):
            if a_mask:
                a = a[mask]

            alpha[mask] = a
            beta[mask] = y[mask] - a * x[mask]

        ##########################
        # Negative Open regions #
        ##########################
        lower_domain = torch.zeros(lower.shape + (2,))
        upper_bisection = torch.zeros(upper.shape + (2,))

        ####  First Quarter (bound > 0 & bound_prime > 0) ####
        at_upper_1st = (inter_act >= 0) & (inter_prime >= 0) & at_upper

        lower_domain[..., 0][at_upper_1st] = -0.5 * self.period
        lower_domain[..., 1][at_upper_1st] = 0. * self.period

        upper_bisection[..., 0][at_upper_1st] = -0.75 * self.period
        upper_bisection[..., 1][at_upper_1st] = -0.5 * self.period

        #### Second Quarter (bound > 0 & bound_prime < 0) ####
        at_upper_2nd = (inter_act >= 0) & (inter_prime < 0) & at_upper

        lower_domain[..., 0][at_upper_2nd] = -0.5 * self.period
        lower_domain[..., 1][at_upper_2nd] = 0. * self.period

        upper_bisection[..., 0][at_upper_2nd] = 0.25 * self.period
        upper_bisection[..., 1][at_upper_2nd] = 0.5 * self.period

        #### Third Quarter (bound < 0 & bound_prime < 0) ####
        at_upper_3th = (inter_act < 0) & (inter_prime <= 0) & at_upper

        lower_domain[..., 0][at_upper_3th] = -0.5 * self.period
        lower_domain[..., 1][at_upper_3th] = 0. * self.period

        upper_bisection[..., 0][at_upper_3th] = 0.25 * self.period
        upper_bisection[..., 1][at_upper_3th] = 0.5 * self.period

        #### Fourth Quarter (bound < 0 & bound_prime > 0) ####
        at_upper_4th = (inter_act < 0) & (inter_prime > 0) & at_upper

        lower_domain[..., 0][at_upper_4th] = 0.75 * self.period
        lower_domain[..., 1][at_upper_4th] = 1.0 * self.period

        upper_bisection[..., 0][at_upper_4th] = 0.25 * self.period
        upper_bisection[..., 1][at_upper_4th] = 0.5 * self.period

        # Construct Bounds
        at_upper_correction = (upper[at_upper] // self.period) * self.period
        lower_domain[..., 0][at_upper] += at_upper_correction
        lower_domain[..., 1][at_upper] += at_upper_correction
        upper_bisection[..., 0][at_upper] += at_upper_correction
        upper_bisection[..., 1][at_upper] += at_upper_correction

        d = self.tangent_strategy.increasing_lower_tangent(
            self, upper[at_upper], lower_domain[..., 0][at_upper], lower_domain[..., 1][at_upper])
        add_linear(self.alpha_lower, self.beta_lower, mask=at_upper, a=self.derivative(d), x=intersection, y=inter_act, a_mask=False)

        d = self.tangent_strategy.decreasing_upper_tangent(
            self, upper[at_upper], upper_bisection[..., 0][at_upper], upper_bisection[...,1][at_upper])
        add_linear(self.alpha_upper, self.beta_upper, mask=at_upper, a=self.derivative(d), x=intersection, y=inter_act, a_mask=False)

        ##########################
        # Positive Open regions #
        ##########################
        lower_domain = torch.zeros(lower.shape + (2,))
        upper_domain = torch.zeros(upper.shape + (2,))

        ### First Quarter (bound > 0 & bound_prime > 0) ###
        at_lower_1st = (inter_act >= 0) & (inter_prime >= 0) & at_lower

        lower_domain[..., 0][at_lower_1st] = 0.5 * self.period
        lower_domain[..., 1][at_lower_1st] = 0.75 * self.period

        upper_domain[..., 0][at_lower_1st] = 0. * self.period
        upper_domain[..., 1][at_lower_1st] = 0.25 * self.period

        #### Second Quarter (bound > 0 & bound_prime < 0) ###
        at_lower_2nd = (inter_act >= 0) & (inter_prime < 0) & at_lower

        lower_domain[..., 0][at_lower_2nd] = 0.5 * self.period
        lower_domain[..., 1][at_lower_2nd] = 0.75 * self.period

        upper_domain[..., 0][at_lower_2nd] = 1.0 * self.period
        upper_domain[..., 1][at_lower_2nd] = 1.25 * self.period

        #### Third Quarter (bound < 0 & bound_prime < 0) ####
        at_lower_3th = (inter_act < 0) & (inter_prime <= 0) & at_lower

        lower_domain[..., 0][at_lower_3th] = 0.5 * self.period
        lower_domain[..., 1][at_lower_3th] = 0.75 * self.period

        upper_domain[..., 0][at_lower_3th] = 1.0 * self.period
        upper_domain[..., 1][at_lower_3th] = 1.25 * self.period

        # Fourth Quarter (bound < 0 & bound_prime > 0)
        at_lower_4th = (inter_act < 0) & (inter_prime > 0) & at_lower

        lower_domain[..., 0][at_lower_4th] = 1.5 * self.period
        lower_domain[..., 1][at_lower_4th] = 1.75 * self.period

        upper_domain[..., 0][at_lower_4th] = 1.0 * self.period
        upper_domain[..., 1][at_lower_4th] = 1.25 * self.period

        # Construct Bounds
        at_lower_correction = (lower[at_lower] // self.period) * self.period
        lower_domain[..., 0][at_lower] += at_lower_correction
        lower_domain[..., 1][at_lower] += at_lower_correction
        upper_domain[..., 0][at_lower] += at_lower_correction
        upper_domain[..., 1][at_lower] += at_lower_correction

        d = self.tangent_strategy.decreasing_lower_tangent(
            self, lower[at_lower], lower_domain[..., 0][at_lower], lower_domain[..., 1][at_lower])
        add_linear(self.alpha_lower, self.beta_lower, mask=at_lower, a=self.derivative(d), x=intersection, y=inter_act, a_mask=False)
        d = self.tangent_strategy.increasing_upper_tangent(
            self, lower[at_lower], upper_domain[..., 0][at_lower], upper_domain[..., 1][at_lower])
        add_linear(self.alpha_upper, self.beta_upper, mask=at_lower, a=self.derivative(d), x=intersection, y=inter_act, a_mask=False)