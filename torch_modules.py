import torch
import bound_propagation as bp

class ScalarMult(torch.nn.Linear): #\todo rename MultScalar
    def __init__(self, in_features: int, scalar: float):
        super(ScalarMult, self).__init__(in_features, in_features, bias=False)
        with torch.no_grad():
            self.weight.copy_(torch.eye(in_features) * scalar)

class ScalarAdd(torch.nn.Linear): # \todo rename AddScalar
    def __init__(self, in_features: int, scalar: float):
        super(ScalarAdd, self).__init__(in_features, in_features)
        with torch.no_grad():
            self.weight.copy_(torch.eye(in_features))
            self.bias.fill_(scalar)

class Sum(torch.nn.Linear):
    def __init__(self, in_features: int):
        super(Sum, self).__init__(in_features, 1, bias=False)
        with torch.no_grad():
            self.weight.fill_(1.0)

class SqNorm(torch.nn.Sequential):
    def __init__(self, num_dims):
        super().__init__(bp.Pow(2), Sum(num_dims))


class BoundSigmoid(bp.BoundSigmoid):
    def __init__(self, *args, **kwargs):
        super(BoundSigmoid, self).__init__(*args, **kwargs)

    @bp.activation.assert_bound_order
    def alpha_beta(self, preactivation):
        lower, upper = preactivation.lower, preactivation.upper
        zero_width, n, p, np = bp.activation.regimes(lower, upper)

        neginf = lower.isneginf()
        inf = upper.isinf()

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
        add_linear(self.alpha_upper, self.beta_upper, mask=n, a=slope, x=upper, y=upper_act)

        # Lower bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_lower, self.beta_lower, mask=n, a=upper_prime, x=upper, y=upper_act)

        # Allow parameterization
        # Save mask
        self.unstable_lower = n
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        self.unstable_d_lower = d[n].detach().clone().requires_grad_()
        # Save ranges to clip (aka. PGD)
        self.unstable_range_lower = lower[n], upper[n]

        ###################
        # Positive regime #
        ###################
        # Lower bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_lower, self.beta_lower, mask=p, a=slope, x=lower, y=lower_act)

        # Upper bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_upper, self.beta_upper, mask=p, a=lower_prime, x=lower, y=lower_act)

        # Allow parameterization
        # Save mask
        self.unstable_upper = p
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        self.unstable_d_upper = d[p].detach().clone().requires_grad_()
        # Save ranges to clip (aka. PGD)
        self.unstable_range_upper = lower[p], upper[p]

        #################
        # Crossing zero #
        #################
        # Upper bound #
        # If tangent to upper is below lower, then take direct slope between lower and upper
        direct = np & (slope <= upper_prime)
        add_linear(self.alpha_upper, self.beta_upper, mask=direct, a=slope, x=lower, y=lower_act)

        # Else use bisection to find upper bound on slope.
        implicit = np & (slope > upper_prime)

        if torch.any(implicit):
            d = self.tangent_strategy.upper_tangent(self, lower.clamp(-10., 0.)[implicit], upper.clamp(0., 10.)[implicit])

            # Slope has to attach to (lower, sigma(lower))
            add_linear(self.alpha_upper, self.beta_upper, mask=implicit, a=self.derivative(d), x=lower, y=lower_act, a_mask=False)

        # Lower bound #
        # If tangent to lower is above upper, then take direct slope between lower and upper
        direct = np & (slope <= lower_prime)
        add_linear(self.alpha_lower, self.beta_lower, mask=direct, a=slope, x=upper, y=upper_act)

        # Else use bisection to find upper bound on slope.
        implicit = np & (slope > lower_prime)

        if torch.any(implicit):
            d = self.tangent_strategy.lower_tangent(self, lower.clamp(-10., 0.)[implicit], upper.clamp(0., 10.)[implicit])

            # Slope has to attach to (upper, sigma(upper))
            add_linear(self.alpha_lower, self.beta_lower, mask=implicit, a=self.derivative(d), x=upper, y=upper_act, a_mask=False)


linear_factory = bp.BoundModelFactory()
linear_factory.register(torch.nn.Sigmoid, BoundSigmoid)