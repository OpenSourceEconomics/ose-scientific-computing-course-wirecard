# In this file we organise all optimization algorithms in a callable format for our performance testing routine.

# They are supposed to take the following format:

# def minimization(f,starting_point, stopping_tolerance_xwert, stopping_tolerance_functionswert, computational _budget):
# ...
# return optimum, number_of_function_evaluations

import numpy as np
from newton_based_optimization_source import naive_optimization

from nelder_mead_based_optimization_source import (
    call_nelder_mead_method,
    initial_simplex,
)


def our_nelder_mead_method(
    f,
    starting_point,
    stopping_tolerance_xvalue,
    stopping_tolerance_functionvalue,
    computational_budget,
    eps=0.5,
):

    eps = np.array([eps] * len(starting_point))
    verts = initial_simplex(
        len(starting_point) + 1, [starting_point - eps, starting_point + eps]
    )
    return call_nelder_mead_method(
        f,
        verts,
        stopping_tolerance_xvalue,
        stopping_tolerance_functionvalue,
        computational_budget,
    )


def our_newton_based_optimization(
    f,
    starting_point,
    stopping_tolerance_xvalue,
    stopping_tolerance_functionvalue,
    computational_budget,
):
    return naive_optimization(
        f,
        starting_point,
        stopping_tolerance_xvalue,
        stopping_tolerance_functionvalue,
        computational_budget,
    )
