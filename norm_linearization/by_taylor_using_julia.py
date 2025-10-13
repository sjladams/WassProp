#import juliacall
# jl = juliacall.newmodule("julias_playground")

from juliacall import Main
import numpy as np
import matplotlib.pyplot as plt
import torch

"""
Implementation of example in Taylor Models documentation via Python
"""


if __name__ == '__main__':
    # Compute Taylor Models in Julia
    Main.seval("""
    using TaylorModels

    c = 2.0
    f(x) = 1 / (1 + exp(-x)) # sigmoid
    g(x) = (f(x) - f(c))^2

    # Define domain and expansion point
    a = 1 .. 2.5

    # Create Taylor models
    tm = TaylorModel1(2, interval(c), a)

    # Compute Taylor models of g(x)
    ftm = g(tm)

    coeffs = [[inf(coef), sup(coef)] for coef in ftm.pol.coeffs]  # coefficient bounds
    rem = [inf(ftm.rem), sup(ftm.rem)]  # remainder bound
    """)

    print(Main.ftm)
    c = torch.tensor(Main.c)

    # Compute Values for Plotting
    X = torch.linspace(1., 2.5, 1000)
    X_plot = torch.norm(X - c, p=2, dim=-1).pow(2)

    # Evaluate f(x) for real numbers in Python
    def f(x):
        return 1 / (1 + np.exp(-x))

    def g(x):
        return (f(x) - f(c))**2

    class Polynomial:
        def __init__(self, coeffs: torch.Tensor, rem: torch.Tensor):
            self.coeffs = coeffs
            self.rem = rem

        def approx(self, x):
            return torch.polynomial.polynomialval(x, self.coeffs)

        def bound(self, x):
            y = self.approx(x)
            return y + self.rem[..., 0], y + self.rem[..., 1]

    Y = g(X)

    polynomial = Polynomial(coeffs=torch.tensor(Main.coeffs).mean(-1).flip(dims=(-1,)), rem=torch.tensor(Main.rem))

    Y_approx = polynomial.approx(X)
    Y_lower, Y_upper = polynomial.bound(X)

    plt.figure(figsize=(8, 5))
    plt.plot(X_plot, Y, label="g(x)", linewidth=2, color="black")
    plt.plot(X_plot, Y_approx, label="approx", color="blue")
    plt.plot(X_plot, Y_lower, label="lower bound", color="red")
    plt.plot(X_plot, Y_upper, label="upper bound", color="green")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()

