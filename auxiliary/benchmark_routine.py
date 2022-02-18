import numpy as np
import pandas as pd


from callable_algorithms import our_nelder_mead_method, our_newton_based_optimization
from functions import rastrigin, griewank, rosenbrock, levi_no_13


def get_starting_points(n, domain_center, domain_radius, dimension):
    A = np.random.rand(n, dimension) * domain_radius
    A = [domain_center + x for x in A]
    return A


def benchmark_optimization(
    f,
    optimum,
    algo,
    computational_budget,
    x_tolerance,
    y_tolerance,
    n,
    domain_center,
    domain_radius,
    dimension,
):

    """Return a dataframe which contains the results of the optimization algorithm applied to the
       test-funcion f with n different starting point.

    Args:
        f:                  a function from \R^n to \R whose optimum we want to find
        optimum:            The correct optimum of f
        algo:               The optimization algorithm we use for the optimization
        computation_budget: the number of maximal function evaluations
        x_tolerance:        a positive real number
        y_tolerance:        a positive real number
        n:                  The number or starting points from which we start the optimization
        domain_center:      The center of the domain in which we want to start the optimization
        domain_radius:      The radius of the domain
        dimension:          The dimension of the problem f


    Returns:
        out:               A dataframe that contains the results and the number of function evaluation for
                           each attemp to optimize f

    """

    starting_points = get_starting_points(n, domain_center, domain_radius, dimension)

    df = pd.DataFrame([], columns=["successfull", "function evaluations"])

    for starting_point in starting_points:
        a = pd.Series(
            algo(f, starting_point, x_tolerance, y_tolerance, computational_budget),
            index=df.index,
        )
        """if (a[0] - optimum) < 1e-6:
            a = a.append(1)
        else:
            a = a.append(0)
        """
        df.append(a, ignore_index=True)

    # for result, function_calls in zip(df)

    return df
    pass


if __name__ == "__main__":

    df1 = benchmark_optimization(
        griewank, [0, 0], our_nelder_mead_method, 100, 1e-4, 1e-6, 100, [0, 0], 5, 2
    )
    df2 = benchmark_optimization(
        griewank,
        [0, 0],
        our_newton_based_optimization,
        100,
        1e-6,
        1e-6,
        100,
        [0, 0],
        5,
        2,
    )
    print(df1)
    print(df2)
    pass
