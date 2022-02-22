import numpy as np
import pandas as pd
import time


from callable_algorithms import our_nelder_mead_method, our_newton_based_optimization
from functions import rastrigin, griewank, rosenbrock, levi_no_13


def get_starting_points(n, domain_center, domain_radius, dimension):
    """Return a dataframe which contains the results of the optimization algorithm applied to the
       test-funcion f with n different starting point.

    Args:
        n:                  the number of points which we want to the function to return
        domain_center:      the center of the domain in which the starting points should lie
        domain_radius:      the radius of the domain
        dimension:          the dimension of the space from which we want the starting point


    Returns:
        out:               returns and array of n starting points within the domain_radius around the domain_center

    """
    A = np.random.rand(n, dimension) * domain_radius
    A = [domain_center + x for x in A]
    return A


def average_time_success(df):
    df_new = pd.DataFrame([])
    df_new["algorithm"] = pd.Series([df.iloc[0]["algorithm"]])
    df_new["test_function"] = pd.Series([df.iloc[0]["test_function"]])
    df_new["computational_budget"] = pd.Series([df.iloc[0]["computational_budget"]])
    df_new["sample_size"] = pd.Series([df.iloc[0]["sample_size"]])
    df_new["success_rate"] = pd.Series([df["success"].mean()])
    df_new["average_time"] = pd.Series([df["time"].mean()])
    df_new["average_function_evaluations"] = pd.Series(
        [df["function_evaluations"].mean()]
    )
    return df_new


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
    algo_name="unknown",
    t_func_name="unknown",
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

    df = pd.DataFrame([], columns=["computed_result", "function_evaluations", "time"])

    for starting_point in starting_points:
        start = time.time()
        a = algo(f, starting_point, x_tolerance, y_tolerance, computational_budget)
        end = time.time()
        a = list(a)
        a = pd.Series(
            a + [end - start],
            index=df.columns,
        )
        df = df.append(a, ignore_index=True)

    df["correct_result"] = pd.Series([optimum] * n)
    df["algorithm"] = pd.Series([algo_name] * n)
    df["test_function"] = pd.Series([t_func_name] * n)
    df["computational_budget"] = pd.Series([computational_budget] * n)
    df["sample_size"] = pd.Series([n] * n)
    df["success"] = df.apply(
        lambda row: np.allclose(row.correct_result, row.computed_result), axis=1
    )
    df["success"] = df.apply(lambda row: row.success * 1, axis=1)

    return df


def run_benchmark(
    algorithm,
    test_function,
    optimum,
    computational_budgets,
    domain_center,
    domain_radius,
    dimension,
    n,
    algorithm_name="unknown",
    test_function_name="unknown",
    x_tolerance=1e-6,
    y_tolerance=1e-6,
):
    df = average_time_success(
        benchmark_optimization(
            test_function,
            optimum,
            algorithm,
            computational_budgets[0],
            x_tolerance,
            y_tolerance,
            n,
            domain_center,
            domain_radius,
            dimension,
            algorithm_name,
            test_function_name,
        )
    )
    count = 0
    for computational_budget in computational_budgets[1:]:
        df = pd.concat(
            [
                df,
                average_time_success(
                    benchmark_optimization(
                        test_function,
                        optimum,
                        algorithm,
                        computational_budget,
                        x_tolerance,
                        y_tolerance,
                        n,
                        domain_center,
                        domain_radius,
                        dimension,
                        algorithm_name,
                        test_function_name,
                    )
                ),
            ],
            axis=0,
        )
        count += 1
        print("Benchmark ", count, " out of ", len(computational_budgets), "done.")
    return df


if __name__ == "__main__":

    test_benchmark_optimization = False
    test_accumulated_benchmark = True

    if test_benchmark_optimization == True:
        df1 = benchmark_optimization(
            griewank,
            [0, 0],
            our_nelder_mead_method,
            100,
            1e-4,
            1e-6,
            100,
            [0, 0],
            5,
            2,
            "our_nelder_mead_method",
            "griewank",
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
        print("df1 summary: \n")
        print(average_time_success(df1))
        print("df2 sumary: \n")
        print(average_time_success(df2))

    if test_accumulated_benchmark == True:
        df = run_benchmark(
            our_nelder_mead_method,
            griewank,
            [0, 0],
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            [0, 0],
            4,
            2,
            25,
            "our nelder mead method",
            "griewank",
        )
        print(df)
    pass
