import numpy as np
import nlopt
from functions import (
    rastrigin,
    rosenbrock,
    griewank,
    levi_no_13,
)
from newton_based_optimization_source import naive_optimization

from nelder_mead_based_optimization_source import (
    call_nelder_mead_method,
    initial_simplex,
)

# In this File we provide three global and two local optimization algorithms.

# 1. local optimization:
#       - our_simple_nelder_mead_method
#       - our_simple_newton_based_optimization

# 2. global optimization
#       - global_optimization_BOBYQA
#       - global_nelder_mead_method
#       - globar_newton_based_optimization

# The global_optimization_BOBYQA uses the optimization algorithm BOBYQA provided by the NLOPT package
# The global_nelder_mead_method uses a implementation of the nelder-mead method that we coded in the file: nelder_mead_based_optimization_source
# the global_newton_based_optimization uses an newton_method_based local optimization we coded in the file: newton_based_optimization_source

# local optimization
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


# local optimization
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


# local optimization
def optimization_BOBYQA_NLOPT(
    f,
    starting_point,
    x_tolerance,
    y_tolerance,
    computational_budget,
):

    # set all the parameters for the optimization algorithm
    n = len(starting_point)
    global_optimum = nlopt.opt(nlopt.GN_MLSL, n)
    local_opt = nlopt.opt(nlopt.LN_BOBYQA, n)
    local_opt.set_xtol_abs(x_tolerance)
    local_opt.set_ftol_abs(y_tolerance)
    local_opt.set_min_objective(f)
    global_optimum.set_local_optimizer(local_opt)
    global_optimum.set_lower_bounds(-100)
    global_optimum.set_upper_bounds(100)
    global_optimum.set_min_objective(f)
    global_optimum.set_xtol_abs(x_tolerance)
    global_optimum.set_ftol_abs(y_tolerance)
    global_optimum.set_maxeval(computational_budget)

    # optimize the function
    optimizer = global_optimum.optimize(np.array(starting_point))

    num_evals = global_optimum.get_numevals()

    # round the result
    optimizer = np.round(optimizer)

    return optimizer, num_evals


# global optimization
def global_nelder_mead_method(
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
    """Return an approximation of a global optimum found by the nelder-mead method run from sample size many starting points.

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


# global optimization
def global_newton_based_optimization(
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
    """Return an approximation of a global optimum found by the newton-method run from multiple starting points.

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


# global optimization
def global_optimization_BOBYQA(
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
    """Return an approximation of a global optimum found by the BOBYQA algorithm run from sample size starting points.

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
        optimization_BOBYQA_NLOPT,
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


# the following function finds the starting points for the global optimization algorithms
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
    # get random vectors in [-1,1]^n
    A = np.random.rand(sample_size, dimension) * 2 - np.array([1] * dimension)
    # map these onto domain_center + [-domain_radius, domain_radius]
    B = [domain_center + x * domain_radius for x in A]
    # apply f onto these points
    C = np.array([f(x) for x in B])
    # find the number_of_candidates points with the smallest f-value
    D = np.argpartition(C, -number_of_candidates)[-number_of_candidates:]
    # return the list of candidates
    return [B[index] for index in D]


# this funciton is used to turn the local searches into global search algorithms
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
    # initilize variables
    budget = computational_budget + 0
    candidates = []
    values = []
    i = 0

    # make sure that the variables computational_budget, sample_size  and number_of_candididates take the right form
    assert (
        computational_budget > sample_size
    ), "computational_budget is initially smaller than sample size."
    assert computational_budget > 0, "computational_budget should be positive."
    assert number_of_candidates > 0, "number_of_candidates should be positive."

    # if we run through all candidates and still have budget left we get new candidates
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
        # while we have budget left we start local searches from the candidates we were given before
        while computational_budget > 0 and i < number_of_candidates:
            a = algorithm(
                f, starting_points[i], x_tolerance, y_tolerance, computational_budget
            )
            candidates.append(a[0])
            values.append(f(a[0]))
            computational_budget -= a[1]
            computational_budget -= 1
            i += 1
    # get the best computed candidate
    index = np.argmin(values)

    return candidates[index], budget - computational_budget


if __name__ == "__main__":
    print(
        "the result ist: ",
        iterate_optimization(
            our_simple_nelder_mead_method, rastrigin, 1000, [0, 0], 5, 2, 50, 5
        ),
    )
    print(optimization_BOBYQA_NLOPT(rosenbrock, [45, 76], 1e-6, 1e-6, 1000))
    print(optimization_BOBYQA_NLOPT(rastrigin, [52, 65], 1e-6, 1e-6, 1000))
    print(optimization_smart_BOBYQA_NLOPT(griewank, 1000, [0, 0], 100, 2))

    # print(optimization_BOBYQA_NLOPT(rastrigin, [75,65],1e-6,1e-6,1000))
    pass
