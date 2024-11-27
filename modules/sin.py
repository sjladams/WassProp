import torch
import bound_propagation as bp

__all__ = ['BoundSin']

class BoundSin(bp.BoundSin):
    def __init__(self, *args, **kwargs):
        super(BoundSin, self).__init__(*args, **kwargs)

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
