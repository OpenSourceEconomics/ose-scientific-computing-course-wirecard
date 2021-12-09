# Until we have a clearer image of how our project is going to unfold,
# I will be commiting my programming-progress in this File

import numpy as np


def first_derivative_1D(x, f, eps=10 ** (-6)):
    """Return an approximation for the derivative of f at x.

    Args:
        x:   a number within the domain of f
        A:   a function that maps a subset of \R to \R
        eps: a scale to controll the accuracy of the approximation


    Returns:
        out: an approximation of the value of the derivative of f at x

    """
    out = (f(x + eps) - f(x - eps)) / eps
    return out


def newton_method_1D(f, x_n, eps_newton=10 ** (-6), eps_derivative=10 ** (-6), n=1000):
    """Return a candidate for a root of f, if the newton method starting at x_n converges.

    Args:
        f:              a function from \R to \R whose root we want to find
        x_n:            a number within the domain of f from which to start the iteration
        eps_newton:     sensitivity of of the root finding process
        eps_derivative: sensitivity of the derivative approximation
        n:              maximum of iterations before stopping the procedure



    Returns:
        out: either an approximation for a root or a message if the procedure didnt converge

    """
    df = lambda a: first_derivative_1D(a, f, eps_derivative)
    for _i in range(1, n):
        x_n = x_n - (f(x_n) / df(x_n))
        if np.abs(f(x_n)) < eps_newton:
            break
    # print("Ran through: ",i, " times.")

    if i > n - 2:
        return "Didnt converge"
    else:
        return x_n
