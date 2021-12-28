# In this file we organise all optimization algorithms in a callable format for our performance testing routine.

# They are supposed to take the following format:

# def minimization(f,starting_point, stopping_tolerance_xwert, stopping_tolerance_functionswert, computational _budget):
# ...
# return optimum, number_of_function_evaluations

from optimization_algorithms_source import nelder_mead_method, naive_optimization


def our_nelder_mead_method(
    f,
    starting_point,
    stopping_tolerance_xvalue,
    stopping_tolerance_functionvalue,
    computational_budget,
):
    pass


def our_newton_based_optimization(
    f,
    starting_point,
    stopping_tolerance_xvalue,
    stopping_tolerance_functionvalue,
    computational_budget,
):
    pass
