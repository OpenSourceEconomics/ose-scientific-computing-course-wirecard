import autograd.numpy as np
import matplotlib.pyplot as plt
import math
import random

from mpl_toolkits.mplot3d import Axes3D
from numpy.lib.function_base import vectorize

# functions to be tested
from auxiliary.test_functions_daniel_copy import rastrigin_instance
from auxiliary.test_functions_daniel_copy import rosenbrock_instance
from auxiliary.test_functions_daniel_copy import levi_no_13_instance
from auxiliary.test_functions_daniel_copy import griewank_instance
from auxiliary.functions_daniel_copy import rastrigin, rosenbrock, levi_no_13, griewank

# from auxiliary.newton_based_optimization_source import (
# find_starting_point,
# naive_optimization,
# newton_method,
# )

# from auxiliary.nelder_mead_based_optimization_source import initial_simplex, nelder_mead_method

# from auxiliary.work_niel_copy import first_derivative_1D, newton_method_1D


def plot_test_function(domain, function, name_of_function="some function"):
    """Plot a 3d graph of a function.

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


def plot_test_function_contour(domain, function, name_of_function, opt):
    """Plot a 3d graph of a function.

    Args:
        domain:                     domain \subset \R of the function
        function:                   a function that returns a value for a input array [a,b] where a,b are in the domain
        name_of_function (string) : the name of the function
        opt: vector that contains coordinates where known optimum lies e.g. (x,y)=(0,0)


    Returns:
        a contour plot of the function

    """
    x_points = np.linspace(domain[0], domain[1], 100)
    y_points = np.linspace(domain[0], domain[1], 100)

    X, Y = np.meshgrid(x_points, y_points)

    rastrigin_vectorized = np.vectorize(lambda a, b: function([a, b]))

    Z = rastrigin_vectorized(X, Y)

    plt.figure(figsize=(7, 7))
    ax = plt.axes()

    plt.contourf(X, Y, Z)
    ax.set(xlabel="x", ylabel="y", title="f = " + name_of_function)
    ax.scatter(opt[0], opt[1], c="red", marker="x")
    plt.show()
    pass
