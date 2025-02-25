import torch
import bound_propagation as bp
import math

__all__ = ['BoundSin']


class SinTangentBisectionStrategy(bp.activation.SinTangentBisectionStrategy):
    def increasing_upper_tangent_inf(self, bound_module, lower, bisection_lower, bisection_upper):
        lower_act = bound_module(lower)

        def f_lower(d: torch.Tensor) -> torch.Tensor:
            a_slope = (bound_module(d)- lower_act) / (d - lower)
            a_derivative = bound_module.derivative(d)
            return a_slope - a_derivative

        _, d_upper = bp.activation.bisection(bisection_lower, bisection_upper, f_lower, num_iter=100)
        return d_upper

    def increasing_lower_tangent_inf(self, bound_module, upper, bisection_lower, bisection_upper):
        upper_act = bound_module(upper)

        def f_lower(d: torch.Tensor) -> torch.Tensor:
            a_slope = (upper_act - bound_module(d)) / (upper - d)
            a_derivative = bound_module.derivative(d)
            return a_derivative - a_slope

        d_lower, _ = bp.activation.bisection(bisection_lower, bisection_upper, f_lower, num_iter=100)
        return d_lower

    def decreasing_upper_tangent_inf(self, bound_module, upper, bisection_lower, bisection_upper):
        upper_act = bound_module(upper)

        def f_upper(d: torch.Tensor) -> torch.Tensor:
            a_slope = (upper_act - bound_module(d)) / (upper - d)
            a_derivative = bound_module.derivative(d)
            return a_slope - a_derivative

        _, d_upper = bp.activation.bisection(bisection_lower, bisection_upper, f_upper, num_iter=100)
        return d_upper

    def decreasing_lower_tangent_inf(self, bound_module, lower, bisection_lower, bisection_upper):
        lower_act = bound_module(lower)

        def f_upper(d: torch.Tensor) -> torch.Tensor:
            a_slope = (bound_module(d) - lower_act) / (d - lower)
            a_derivative = bound_module.derivative(d)
            return a_derivative - a_slope

        d_lower, _ = bp.activation.bisection(bisection_lower, bisection_upper, f_upper, num_iter=100)
        return d_lower


class BoundSin(bp.BoundSin):
    def __init__(self, *args, **kwargs):
        super(BoundSin, self).__init__(*args, **kwargs)
        self.tangent_strategy = kwargs.get('sin_tangent_strategy', SinTangentBisectionStrategy())

    @bp.activation.assert_bound_order
    def alpha_beta(self, preactivation):
        lower, upper = preactivation.lower, preactivation.upper

        zero_width, half_period, (_, increasing), (_, decreasing), crossing_peak, crossing_trough = \
            bp.activation.sine_like_regimes(lower, upper, period=self.period, zero_increasing=self.zero_increasing)
        increasing_lower_curve, increasing_upper_curve, increasing_full_region = increasing
        decreasing_lower_curve, decreasing_upper_curve, decreasing_full_region = decreasing

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        # Use upper and lower in the bias to account for a small numerical difference between lower and upper
        # which ought to be negligible, but may still be present due to torch.isclose.
        self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, self(lower[zero_width])
        self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, self(upper[zero_width])

        lower_act, upper_act = self(lower), self(upper)
        lower_prime, upper_prime = self.derivative(lower), self.derivative(upper)

        d = (lower + upper) * 0.5  # Let d be the midpoint of the two bounds
        d_act = self(d)
        d_prime = self.derivative(d)

        slope = (upper_act - lower_act) / (upper - lower)

        ones = torch.ones_like(lower)
        zeros = torch.zeros_like(lower)

        def add_linear(alpha, beta, mask, a, x, y, a_mask=True):
            if a_mask:
                a = a[mask]

            alpha[mask] = a
            beta[mask] = y[mask] - a * x[mask]

        ##################
        # >= Half period #
        ##################
        # Lower bound
        # - Flat line = -1
        add_linear(self.alpha_lower, self.beta_lower, mask=half_period, a=zeros, x=zeros, y=-ones)

        # Upper bound
        # - Flat line = +1
        add_linear(self.alpha_upper, self.beta_upper, mask=half_period, a=zeros, x=zeros, y=ones)

        ###############
        # Lower curve #
        ###############
        lower_curve = increasing_lower_curve | decreasing_lower_curve | crossing_trough

        # Upper bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_upper, self.beta_upper, mask=lower_curve, a=slope, x=lower, y=lower_act)

        # Lower bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_lower, self.beta_lower, mask=lower_curve, a=d_prime, x=d, y=d_act)

        # Allow parameterization
        # Save mask
        self.unstable_lower = lower_curve
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        self.unstable_d_lower = d[lower_curve].detach().clone().requires_grad_()
        # Save ranges to clip (aka. PGD)
        self.unstable_range_lower = lower[lower_curve], upper[lower_curve]

        ###############
        # Upper curve #
        ###############
        upper_curve = increasing_upper_curve | decreasing_upper_curve | crossing_peak

        # Lower bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_lower, self.beta_lower, mask=upper_curve, a=slope, x=upper, y=upper_act)

        # Upper bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_upper, self.beta_upper, mask=upper_curve, a=d_prime, x=d, y=d_act)

        # Allow parameterization
        # Save mask
        self.unstable_upper = upper_curve
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        self.unstable_d_upper = d[upper_curve].detach().clone().requires_grad_()
        # Save ranges to clip (aka. PGD)
        self.unstable_range_upper = lower[upper_curve], upper[upper_curve]

        # ##########################
        # Increasing full region #
        ##########################
        # Upper bound #
        # If tangent to upper is below lower, then take direct slope between lower and upper
        direct = increasing_full_region & (slope <= upper_prime)
        add_linear(self.alpha_upper, self.beta_upper, mask=direct, a=slope, x=lower, y=lower_act)

        # Else use bisection to find upper bound on slope.
        implicit = increasing_full_region & (slope > upper_prime)

        if torch.any(implicit):
            d = self.tangent_strategy.increasing_upper_tangent(self, lower[implicit], upper[implicit])

            # Slope has to attach to (lower, sigma(lower))
            add_linear(self.alpha_upper, self.beta_upper, mask=implicit, a=self.derivative(d), x=lower, y=lower_act, a_mask=False)

        # Lower bound #
        # If tangent to lower is above upper, then take direct slope between lower and upper
        direct = increasing_full_region & (slope <= lower_prime)
        add_linear(self.alpha_lower, self.beta_lower, mask=direct, a=slope, x=upper, y=upper_act)

        # Else use bisection to find upper bound on slope.
        implicit = increasing_full_region & (slope > lower_prime)

        if torch.any(implicit):
            d = self.tangent_strategy.increasing_lower_tangent(self, lower[implicit], upper[implicit])

            # Slope has to attach to (upper, sigma(upper))
            add_linear(self.alpha_lower, self.beta_lower, mask=implicit, a=self.derivative(d), x=upper, y=upper_act, a_mask=False)

        ##########################
        # Decreasing full region #
        ##########################
        # Upper bound #
        # If tangent to lower is below upper, then take direct slope between lower and upper
        direct = decreasing_full_region & (slope >= lower_prime)
        add_linear(self.alpha_upper, self.beta_upper, mask=direct, a=slope, x=lower, y=lower_act)

        # Else use bisection to find upper bound on slope.
        implicit = decreasing_full_region & (slope < lower_prime)

        if torch.any(implicit):
            d = self.tangent_strategy.decreasing_upper_tangent(self, lower[implicit], upper[implicit])

            # Slope has to attach to (lower, sigma(lower))
            add_linear(self.alpha_upper, self.beta_upper, mask=implicit, a=self.derivative(d), x=upper, y=upper_act, a_mask=False)

        # Lower bound #
        # If tangent to upper is above lower, then take direct slope between lower and upper
        direct = decreasing_full_region & (slope >= upper_prime)
        add_linear(self.alpha_lower, self.beta_lower, mask=direct, a=slope, x=lower, y=lower_act)

        # Else use bisection to find upper bound on slope.
        implicit = decreasing_full_region & (slope < upper_prime)

        if torch.any(implicit):
            d = self.tangent_strategy.decreasing_lower_tangent(self, lower[implicit], upper[implicit])

            # Slope has to attach to (upper, sigma(upper))
            add_linear(self.alpha_lower, self.beta_lower, mask=implicit, a=self.derivative(d), x=lower, y=lower_act, a_mask=False)

        ##########################
        # Negative Open regions #
        ##########################
        neginf = lower.isneginf() & ~upper.isinf()

        lower_bisection_lower, lower_bisection_upper = torch.zeros_like(lower), torch.zeros_like(lower)
        upper_bisection_lower, upper_bisection_upper = torch.zeros_like(upper), torch.zeros_like(upper)

        ####  First Quarter (bound > 0 & bound_prime > 0) ####
        neginf_1st = (upper_act > 0) & (upper_prime > 0) & neginf

        lower_bisection_lower[neginf_1st] = -0.5 * self.period
        lower_bisection_upper[neginf_1st] = 0. * self.period

        upper_bisection_lower[neginf_1st] = -0.75 * self.period
        upper_bisection_lower[neginf_1st] = -0.5 * self.period

        #### Second Quarter (bound > 0 & bound_prime < 0) ####
        neginf_2nd = (upper_act > 0) & (upper_prime < 0) & neginf

        lower_bisection_lower[neginf_2nd] = -0.5 * self.period
        lower_bisection_upper[neginf_2nd] = 0. * self.period

        upper_bisection_lower[neginf_2nd] = 0.25 * self.period
        upper_bisection_upper[neginf_2nd] = 0.5 * self.period

        #### Third Quarter (bound < 0 & bound_prime < 0) ####
        neginf_3th = (upper_act < 0) & (upper_prime < 0) & neginf

        lower_bisection_lower[neginf_3th] = -0.5 * self.period
        lower_bisection_upper[neginf_3th] = 0. * self.period

        upper_bisection_lower[neginf_3th] = 0.25 * self.period
        upper_bisection_upper[neginf_3th] = 0.5 * self.period

        #### Fourth Quarter (bound < 0 & bound_prime > 0) ####
        neginf_4th = (upper_act < 0) & (upper_prime > 0) & neginf

        lower_bisection_lower[neginf_4th] = 0.75 * self.period
        lower_bisection_upper[neginf_4th] = 1.0 * self.period

        upper_bisection_lower[neginf_4th] = 0.25 * self.period
        upper_bisection_upper[neginf_4th] = 0.5 * self.period

        # Construct Bounds
        neginf_correction = (upper[neginf] // self.period) * self.period
        lower_bisection_lower[neginf] += neginf_correction
        lower_bisection_upper[neginf] += neginf_correction
        upper_bisection_lower[neginf] += neginf_correction
        upper_bisection_upper[neginf] += neginf_correction

        d = self.tangent_strategy.increasing_lower_tangent_inf(
            self, upper[neginf], lower_bisection_lower[neginf], lower_bisection_upper[neginf])
        add_linear(self.alpha_lower, self.beta_lower, mask=neginf, a=self.derivative(d), x=upper, y=upper_act, a_mask=False)

        d = self.tangent_strategy.decreasing_upper_tangent_inf(
            self, upper[neginf], upper_bisection_lower[neginf], upper_bisection_upper[neginf])
        add_linear(self.alpha_upper, self.beta_upper, mask=neginf, a=self.derivative(d), x=upper, y=upper_act, a_mask=False)

        ##########################
        # Positive Open regions #
        ##########################
        inf = ~lower.isneginf() & upper.isinf()

        ### First Quarter (bound > 0 & bound_prime > 0) ###
        inf_1st = (lower_act > 0) & (lower_prime > 0) & inf

        lower_bisection_lower[inf_1st] = 0.5 * self.period
        lower_bisection_upper[inf_1st] = 0.75 * self.period

        upper_bisection_lower[inf_1st] = 0. * self.period
        upper_bisection_upper[inf_1st] = 0.25 * self.period

        #### Second Quarter (bound > 0 & bound_prime < 0) ###
        inf_2nd = (lower_act > 0) & (lower_prime < 0) & inf

        lower_bisection_lower[inf_2nd] = 0.5 * self.period
        lower_bisection_upper[inf_2nd] = 0.75 * self.period

        upper_bisection_lower[inf_2nd] = 1.0 * self.period
        upper_bisection_upper[inf_2nd] = 1.25 * self.period

        #### Third Quarter (bound < 0 & bound_prime < 0) ####
        inf_3th = (lower_act < 0) & (lower_prime < 0) & inf

        lower_bisection_lower[inf_3th] = 0.5 * self.period
        lower_bisection_upper[inf_3th] = 0.75 * self.period

        upper_bisection_lower[inf_3th] = 1.0 * self.period
        upper_bisection_upper[inf_3th] = 1.25 * self.period

        # Fourth Quarter (bound < 0 & bound_prime > 0)
        inf_4th = (lower_act < 0) & (lower_prime > 0) & inf

        lower_bisection_lower[inf_4th] = 1.5 * self.period
        lower_bisection_upper[inf_4th] = 1.75 * self.period

        upper_bisection_lower[inf_4th] = 1.0 * self.period
        upper_bisection_upper[inf_4th] = 1.25 * self.period

        # Construct Bounds
        inf_correction = (lower[inf] // self.period) * self.period
        lower_bisection_lower[inf] += inf_correction
        lower_bisection_upper[inf] += inf_correction
        upper_bisection_lower[inf] += inf_correction
        upper_bisection_upper[inf] += inf_correction

        d = self.tangent_strategy.decreasing_lower_tangent_inf(
            self, lower[inf], lower_bisection_lower[inf], lower_bisection_upper[inf])
        add_linear(self.alpha_lower, self.beta_lower, mask=inf, a=self.derivative(d), x=lower, y=lower_act, a_mask=False)
        d = self.tangent_strategy.increasing_upper_tangent_inf(
            self, lower[inf], upper_bisection_lower[inf], upper_bisection_upper[inf])
        add_linear(self.alpha_upper, self.beta_upper, mask=inf, a=self.derivative(d), x=lower, y=lower_act, a_mask=False)