import numpy as np
from functions import rosenbrock
from functions import griewank
from newton_based_optimization_source import naive_optimization

from nelder_mead_based_optimization_source import (
    call_nelder_mead_method,
    initial_simplex,
)

# In this File we provide two versions


def our_simple_nelder_mead_method(
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
        out_1: an approximation of a local optimum of the function
        out_2: number of evaluations of f

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


def our_simple_newton_based_optimization(
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
        out_1: an approximation of a local optimum of the function
        out_2: number of evaluations of f

    """
    return naive_optimization(
        f,
        starting_point,
        stopping_tolerance_xvalue,
        stopping_tolerance_functionvalue,
        computational_budget,
    )


def our_smart_nelder_mead_method(
    f,
    computational_budget,
    domain_center,
    domain_radius,
    dimension,
    sample_size=50,
    number_of_candidates=5,
    x_tolerance=1e-6,
    y_tolerance=1e-6,
):
    """Return an approximation of a local optimum found by the nelder-mead method run from 50 starting points.

    Args:
        f:                                  a real valued function
        computational_budget:               maximal number of function calls after which the algortithm terminates
        domain_center:                      the center of the domain(-circle)
        domain_radius:                      the radius of the domain(-circle)
        sample_size:                        the number of points which we consider to start the nelder-mead-method from
        number_of_candidates:               the number of points from which we start the nelder-mead-method
        x_tolerance:                        a positive real number
        y_tolerance:                        a positive real number

    Returns:
        out_1: an approximation of a local optimum of the function
        out_2: number of evaluations of f

    """
    return iterate_optimization(
        our_simple_nelder_mead_method,
        f,
        computational_budget,
        domain_center,
        domain_radius,
        dimension,
        sample_size,
        number_of_candidates,
        x_tolerance,
        y_tolerance,
    )


def our_smart_newton_based_optimization(
    f,
    computational_budget,
    domain_center,
    domain_radius,
    dimension,
    sample_size=50,
    number_of_candidates=5,
    x_tolerance=1e-6,
    y_tolerance=1e-6,
):
    """Return an approximation of a local optimum found by the newton-method run from multiple.

    Args:
        f:                                  a real valued function
        computational_budget:               maximal number of function calls after which the algortithm terminates
        domain_center:                      the center of the domain(-circle)
        domain_radius:                      the radius of the domain(-circle)
        sample_size:                        the number of points which we consider to start the newton-method from
        number_of_candidates:               the number of points from which we start the nelder-mead-method
        x_tolerance:                        a positive real number
        y_tolerance:                        a positive real number

    Returns:
        out_1: an approximation of a local optimum of the function
        out_2: number of evaluations of f

    """
    return iterate_optimization(
        our_simple_newton_based_optimization,
        f,
        computational_budget,
        domain_center,
        domain_radius,
        dimension,
        sample_size,
        number_of_candidates,
        x_tolerance,
        y_tolerance,
    )


def find_starting_points(
    f, domain_center, domain_radius, dimension, sample_size, number_of_candidates
):
    """Returns candidates to start the local optimum finding process from.
    Args:
        f:                    a function from \R^n to \R whose optimum we want to find
        domain_center:        the center of the domain(-circle)
        domain_radius:        the radius of the domain(-circle)
        dimension:            the dimension of the inputs of f
        sample_size:          the number of points within the domain to be considered for the optimization
        number_of_candidates: the number of points to be returned as candidates for the optimization to start from

    Returns:
        out:                  an array of points within the domain(-circle)

    """
    A = np.random.rand(sample_size, dimension) * 2 - np.array([1] * dimension)
    B = [domain_center + x * domain_radius for x in A]
    C = np.array([f(x) for x in B])
    D = np.argpartition(C, -number_of_candidates)[-number_of_candidates:]
    return [B[index] for index in D]


def iterate_optimization(
    algorithm,
    f,
    computational_budget,
    domain_center,
    domain_radius,
    dimension,
    sample_size=50,
    number_of_candidates=5,
    x_tolerance=1e-6,
    y_tolerance=1e-6,
):
    """Return an approximation of a local optimum found an algorithm run from multiple starting points.

    Args:
        algorithm,                          an optimization algorithm that takes the inputs: f, starting_point, x_tolerance, y_tolerance, computational_budget
        f:                                  a real valued function
        computational_budget:               maximal number of function calls after which the algortithm terminates
        domain_center:                      the center of the domain(-circle)
        domain_radius:                      the radius of the domain(-circle)
        dimension:                          the dimension of the inputs of f
        sample_size:                        the number of points which we consider to start the nelder-mead-method from
        number_of_candidates:               the number of points from which we start the nelder-mead-method
        x_tolerance:                        a positive real number
        y_tolerance:                        a positive real number

    Returns:
        out_1: an approximation of a local optimum of the function
        out_2: number of evaluations of f

    """
    budget = computational_budget + 0
    candidates = []
    values = []

    i = 0
    while computational_budget > sample_size:
        starting_points = find_starting_points(
            f,
            domain_center,
            domain_radius,
            dimension,
            sample_size,
            number_of_candidates,
        )
        computational_budget -= sample_size
        i = 0
        while computational_budget > 0 and i < number_of_candidates:
            a = algorithm(
                f, starting_points[i], x_tolerance, y_tolerance, computational_budget
            )
            candidates.append(a[0])
            values.append(f(a[0]))
            computational_budget -= a[1]
            computational_budget -= 1
            i += 1
    index = np.argmin(values)

    return candidates[index], budget - computational_budget


if __name__ == "__main__":
    print(
        iterate_optimization(
            our_simple_nelder_mead_method, rosenbrock, 1000, [0, 0], 5, 2, 50, 5
        )
    )
    pass
