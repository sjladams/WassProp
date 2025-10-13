import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def taylor_expansion(f, x_sym, x0, n):
    """
    Compute the Taylor series expansion of f at x0 up to order n.

    Parameters:
    f  : sympy expression (function of x)
    x  : sympy symbol (variable)
    x0 : point around which to expand
    n  : order of expansion

    Returns:
    sympy expression representing the Taylor series
    """
    return sp.series(f, x_sym, x0, n + 1).removeO()


if __name__ == '__main__':
    n = 2
    c = 4.0
    domain = [c-10., c+10]

    # Define the function and variable
    x_sym = sp.symbols('x')
    # f = sp.exp(x_sym)
    f = 1 / (1 + sp.exp(-x_sym))
    # f = sp.sin(x_sym)
    # f = -x_sym**2 + 0.1*x_sym**4

    # g = sp.Pow(f -  f.subs(x_sym, c), 2)
    g = (f - f.subs(x_sym, c)) ** 2
    g = g.simplify()
    g_func = sp.lambdify(x_sym, g, 'numpy')

    taylor_series = taylor_expansion(g, x_sym, x0=c, n=n)
    taylor_func = sp.lambdify(x_sym, taylor_series, 'numpy')
    coeff_dict = {int(term.as_base_exp()[1]) if term != 1 else 0: coeff for term, coeff in
              taylor_series.as_coefficients_dict().items()}

    def norm_bound(x):
        y = 0.
        for i, coeff in coeff_dict.items():
            y += abs(coeff) * abs(x - c) ** i
        return y

    # Compute Values for Plotting
    X = np.linspace(*domain, 1000)
    X_plot = (X - c) ** 2
    Y = g_func(X)

    Y_approx = taylor_func(X)
    Y_bound = norm_bound(X)

    plt.figure(figsize=(5, 4))
    plt.plot(X_plot, Y, label="true", linewidth=2, color="black")
    plt.plot(X_plot, Y_approx, label="taylor", color="blue")
    plt.plot(X_plot, Y_bound, label="bound", color="red",linestyle="-.")
    plt.xlabel("$||x - c||^2$")
    plt.ylabel("$||f(x)-f(c)||^2$")
    plt.title(f"f(x) = {f}, c={c}")
    plt.legend()
    plt.grid()
    plt.tight_layout()  # Adjust layout to fit labels
    plt.show()