# In this file we organise all optimization algorithms in a callable format for our performance testing routine.

# They are supposed to take the following format:

# def minimization(f,starting_point, stopping_tolerance_xwert, stopping_tolerance_functionswert, computational _budget):
# ...
# return optimum, number_of_function_evaluations

import numpy as np
from newton_based_optimization_source import naive_optimization

from nelder_mead_based_optimization_source import(
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
    """Return an approximation of a local optimum found by the nelder-mead method.

    Args:
        f:                                  a real valued function
        starting point:                     a point within the domain of f around which the approximation starts
        stopping_tolerance_xvalue:          the tolerance of the stopping criterion in the x argument
        stopping_tolerance_functionvalue:   the tolerance of the stopping criterion in the function value
        computational_budget:               maximal number of function calls after which the algortithm terminates 
        eps:                                a measure to control the size of the inital simplex


    Returns:
        out: an approximation of a local optimum of the function, number of evaluations of f 

    """
    eps = np.array([eps] * len(starting_point))
    verts = initial_simplex(
        len(starting_point), [starting_point - eps, starting_point + eps]
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
    """Return an approximation of a local optimum.

        Args:
            f:                                  a real valued function
            starting point:                     a point within the domain of f around which the approximation starts
            stopping_tolerance_xvalue:          the tolerance of the stopping criterion in the x argument
            stopping_tolerance_functionvalue:   the tolerance of the stopping criterion in the function value
            computational_budget:               maximal number of function calls after which the algortithm terminates 
            
        Returns:
            out: an approximation of a local optimum of the function, number of evaluations of f 

        """
    return naive_optimization(
        f,
        starting_point,
        stopping_tolerance_xvalue,
        stopping_tolerance_functionvalue,
        computational_budget,
    )

if __name__ == "__main__": 
    pass