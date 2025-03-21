import torch
import bound_propagation as bp
from .utils import is_vertice

__all__ = ['BoundSigmoid']

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
        zero_width, n, p, np = bp.activation.regimes(lower, upper)

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        # Use upper and lower in the bias to account for a small numerical difference between lower and upper
        # which ought to be negligible, but may still be present due to torch.isclose.
        self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, self(lower[zero_width])
        self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, self(upper[zero_width])

        lower_prime, upper_prime = self.derivative(lower), self.derivative(upper)

        inter_act = self(intersection)
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