import autograd.numpy as np
import matplotlib.pyplot as plt
import math
import random

from mpl_toolkits.mplot3d import Axes3D
from numpy.lib.function_base import vectorize

# functions to be tested
from auxiliary.functions import rastrigin, rosenbrock, levi_no_13, griewank

from newton_based_optimization_source import (
    naive_optimization,
    newton_method,
)

from nelder_mead_based_optimization_source import (
    initial_simplex,
    call_nelder_mead_method,
)


nelder_mead_method = call_nelder_mead_method


def plot_test_function(domain, function, name_of_function="some function"):
    """Plot a 3d graph of a 2-dimensional real valued function.

    Args:
        domain:                     domain \subset \R of the function
        function:                   a function that returns a value for a input array [a,b] where a,b are in the domain
        name_of_function (string) : the name of the function


    Returns:
        a 3d plot of the function

    """
    x_points = np.linspace(domain[0], domain[1], 100)
    y_points = np.linspace(domain[0], domain[1], 100)

    X, Y = np.meshgrid(x_points, y_points)

    rastrigin_vectorized = np.vectorize(lambda a, b: function([a, b]))

    Z = rastrigin_vectorized(X, Y)

    plt.figure(figsize=(20, 10))
    ax = plt.axes(projection="3d")

    ax.plot_surface(X, Y, Z)
    ax.set(xlabel="x", ylabel="y", zlabel="f(x, y)", title="f = " + name_of_function)
    plt.show()
    pass


def critical_function(a):
    print("input to critical function equals: ", a)
    out = (1 / 200) * (a[0] + 1) ** 2 * (math.cos(a[1]) + 1) + a[1] ** 2
    return out


if __name__ == "__main__":

    test_critical_function = False
    plot_test_functions = False
    test_newton_1D = False
    test_newton = True
    test_nelder_mead = False
    test_initial_simplex = False

    if test_critical_function:
        inputs = [np.array([1, 2]), np.array([7, 6]), np.array([np.pi, 2])]
        for input in inputs:
            print(
                "critical function evaluated at some point equals: ",
                critical_function(input),
            )

    if test_initial_simplex:
        print(initial_simplex(2, [-10, 10]))

    if test_nelder_mead:
        # functions whose minimum we want to find:
        f = lambda a: (a[0] + a[1]) ** 2 - 0.5 * a[1] + 1
        print(
            " x -> (x1 + x2)^2 - 0.5 x_2 + 1 hat ein optimum bei: ",
            nelder_mead_method(f, initial_simplex(2, [-10, 10]), 2),
        )

    if plot_test_functions:
        plot_test_function((-5.12, 5.12), rastrigin, "rastrigin")
        plot_test_function((-100, 100), griewank, "griewank")
        plot_test_function((-10, 10), levi_no_13, "Levi no. 13")
        plot_test_function((-100, 100), rosenbrock, "rosenbrock")

    if test_newton_1D:
        print(
            "One root of x^2 + 1 is at x == ", newton_method_1D(lambda a: a ** 2 - 4, 6)
        )
        f = lambda a: (a - 4) ** 2 + 1
        df_2 = lambda b: 2 * b - 8
        df = lambda c: first_derivative_1D(c, f, eps=10 ** -8)
        print("Min of (x-4)^2 + 1 is at x == ", newton_method_1D(df_2, 1))
        print("Min of (x-4)**2 + 1 is at x == ", newton_method_1D(df, 1))

    if test_newton:
        # functions whose minimum we want to find:
        f = lambda a: (a[0] + a[1]) ** 2 - 0.5 * a[1] ** 2 + 1
        # thus we want to find the zeros of its derivative:
        df = lambda b: np.array([2 * b[0] + 2 * b[1] + 1, 2 * b[0] + b[1]])
        # its derivatives Jacobian is given by:
        J = lambda c: np.array([[2, 2], [2, 1]])
        print(
            "x -> [2 * x_1 + 2 * x_2 + 1, 2 * x_1 + x_2] hat eine Nulstelle bei: ",
            newton_method(df, J, np.array([20.234, 100.391])),
        )
        print(
            "x -> (x_1 + x_2)**2 - 0.5 * x_2**2 + 1 hat ein minimum bei: ",
            naive_optimization(f, 2, np.array([1, 1])),
        )

        # f_1 = lambda a: (1 / 200) * (a[0] + 1) ** 2 * (math.cos(a[1]) + 1) + a[1] ** 2
        # f_1 = vectorize(critical_function)
        print("vectorized")
        print(
            "The critical function has a minimum at: ",
            naive_optimization(critical_function, 2, np.array([3, 3])),
        )

        f_2 = lambda a: a  # not in the mood to solve this differential equation
        df_2 = lambda a: np.array(
            [
                a[0] ** 2 - a[1] + a[0] * np.cos(np.pi * a[0]),
                a[0] * a[1] + math.exp(-a[1]) - 1 / a[0],
            ]
        )
        J_2 = lambda a: np.array(
            [
                [
                    2 * a[0]
                    + np.cos(np.pi * a[0])
                    - np.pi * a[0] * np.sin(np.pi * a[0]),
                    -1,
                ],
                [a[1] + 1 / (a[0] ** 2), a[0] - math.exp(-a[1])],
            ]
        )

        print(
            "df_2 hat eine Nulstelle bei: ", newton_method(df_2, J_2, np.array([2, 1]))
        )

        print(
            " x -> (x1 + x2)^2 - 0.5 x_2^2 + 1 hat ein optimum bei: ",
            naive_optimization(f, 2, np.array([-10, 10])),
        )

        f_3 = lambda a: rosenbrock(a)  # n - dimensionale rosenbrock
        print(
            "Die 30-dim Rosenbrock funktion hat ein Optimum bei: ",
            naive_optimization(
                f_3,
                12,
                np.array(
                    [
                        89.0,
                        21.0,
                        43.0,
                        55.0,
                        12.0,
                        13.0,
                        89.0,
                        21.0,
                        43.0,
                        55.0,
                        12.0,
                        13.0,
                        89.0,
                        21.0,
                        43.0,
                        55.0,
                        12.0,
                        13.0,
                        89.0,
                        21.0,
                        43.0,
                        55.0,
                        12.0,
                        13.0,
                        89.0,
                        21.0,
                        43.0,
                        55.0,
                        12.0,
                        13.0,
                    ]
                ),
            ),
        )

        f_4 = lambda a: griewank(a)  # 2 dimensionale griewank
        print(
            "Die 2-dim griewank funktion hat ein Optimum bei: ",
            naive_optimization(f_4, 2, [-5, 5]),
        )

        f_5 = lambda a: rastrigin(a)  # 2 dimensional rastrigin
        print(
            "Die 2-dim rastrigin function hat ein Optimum bei: ",
            naive_optimization(f_5, 2, [-5, 5]),
        )
