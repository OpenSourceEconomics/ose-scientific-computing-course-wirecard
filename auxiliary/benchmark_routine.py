import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


from callable_algorithms import (
    our_simple_nelder_mead_method,
    our_simple_newton_based_optimization,
    our_smart_nelder_mead_method,
    our_smart_newton_based_optimization,
)
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


def benchmark_smart_optimization(
    f,
    algorithm,
    computational_budget,
    domain_center,
    domain_radius,
    number_of_optimizations,
    dimension,
    optimum,
    algo_name="unknown",
    t_func_name="unknown",
    x_tolerance=1e-6,
    y_tolerance=1e-6,
    sample_size=50,
    number_of_candidates=5,
):
    df = pd.DataFrame([], columns=["computed_result", "function_evaluations", "time"])

    for i in range(number_of_optimizations):
        start = time.time()
        a = algorithm(
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
        end = time.time()
        a = list(a)
        a = pd.Series(
            a + [end - start],
            index=df.columns,
        )
        df = df.append(a, ignore_index=True)

    df["correct_result"] = pd.Series([optimum] * number_of_optimizations)
    df["algorithm"] = pd.Series([algo_name] * number_of_optimizations)
    df["test_function"] = pd.Series([t_func_name] * number_of_optimizations)
    df["computational_budget"] = pd.Series(
        [computational_budget] * number_of_optimizations
    )
    df["sample_size"] = pd.Series([number_of_optimizations] * number_of_optimizations)
    df["success"] = df.apply(
        lambda row: np.allclose(row.correct_result, row.computed_result), axis=1
    )
    df["success"] = df.apply(lambda row: row.success * 1, axis=1)

    return df
    pass


def benchmark_simple_optimization(
    f,
    optimum,
    algo,
    computational_budget,
    n,
    domain_center,
    domain_radius,
    dimension,
    algo_name="unknown",
    t_func_name="unknown",
    x_tolerance=1e-6,
    y_tolerance=1e-6,
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


def run_simple_benchmark(
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
        benchmark_simple_optimization(
            test_function,
            optimum,
            algorithm,
            computational_budgets[0],
            n,
            domain_center,
            domain_radius,
            dimension,
            algorithm_name,
            test_function_name,
            x_tolerance,
            y_tolerance,
        )
    )
    count = 0
    print("\n \n Hallo \n\n")
    for computational_budget in computational_budgets[1:]:
        df = pd.concat(
            [
                df,
                average_time_success(
                    benchmark_simple_optimization(
                        test_function,
                        optimum,
                        algorithm,
                        computational_budget,
                        n,
                        domain_center,
                        domain_radius,
                        dimension,
                        algorithm_name,
                        test_function_name,
                        x_tolerance,
                        y_tolerance,
                    )
                ),
            ],
            axis=0,
        )
        count += 1
        print("Benchmark ", count, " out of ", len(computational_budgets), "done.")
    return df


def run_smart_benchmark(
    algorithm,
    test_function,
    optimum,
    computational_budgets,
    domain_center,
    domain_radius,
    dimension,
    number_of_optimizations,
    sample_size=100,
    number_of_candidates=5,
    algorithm_name="unknown",
    test_function_name="unknown",
    x_tolerance=1e-6,
    y_tolerance=1e-6,
):

    df = average_time_success(
        benchmark_smart_optimization(
            test_function,
            algorithm,
            computational_budgets[0],
            domain_center,
            domain_radius,
            number_of_optimizations,
            dimension,
            optimum,
            algorithm_name,
            test_function_name,
            x_tolerance,
            y_tolerance,
            sample_size,
            number_of_candidates,
        )
    )
    count = 0
    for computational_budget in computational_budgets[1:]:
        # print(computational_budget)
        df = pd.concat(
            [
                df,
                average_time_success(
                    benchmark_smart_optimization(
                        test_function,
                        algorithm,
                        computational_budget,
                        domain_center,
                        domain_radius,
                        number_of_optimizations,
                        dimension,
                        optimum,
                        algorithm_name,
                        test_function_name,
                        x_tolerance,
                        y_tolerance,
                        sample_size,
                        number_of_candidates,
                    )
                ),
            ],
            axis=0,
        )
        count += 1
        print("Benchmark ", count, " out of ", len(computational_budgets), "done.")
    return df


def plot_benchmark_results(df):
    df.plot(x="computational_budget", y="success_rate", kind="line")
    plt.show()


if __name__ == "__main__":

    test_benchmark_optimization = False
    test_accumulated_benchmark = True

    input_functions = [
        lambda a: 20 * (a[0] + a[1]) ** 2 + a[1] ** 4 + 1,
        lambda a: (1 / 200) * (a[0] + 1) ** 2 * (np.cos(a[1]) + 1) + a[1] ** 2,
        lambda a: (1 / 800) * (a[0] - 6) ** 4 * (np.sin(a[1]) + 3) + a[1] ** 4,
    ]

    input_function_names = [
        "lambda a: 20 * (a[0] + a[1]) ** 2 + a[1] ** 4 + 1",
        "lambda a: (1 / 200) * (a[0] + 1) ** 2 * (np.cos(a[1]) + 1) + a[1] ** 2",
        "lambda a : (1/800) * ( a[0] - 6) ** 4 * (np.sin(a[1])+ 3) + a[1] ** 4",
    ]

    expected_minima = [[0, 0], [-1, 0], [6, 0]]

    if test_benchmark_optimization == True:
        df1 = benchmark_smart_optimization(
            griewank,
            our_smart_nelder_mead_method,
            400,
            [0, 0],
            5,
            25,
            2,
            [0, 0],
            "our smart nelder mead method",
            "griewank",
        )
        df2 = benchmark_smart_optimization(
            griewank,
            our_smart_newton_based_optimization,
            400,
            [0, 0],
            5,
            25,
            2,
            [0, 0],
            "our smart newton based optimization",
            "griewank",
        )
        print("df1 summary: \n")
        print(average_time_success(df1))
        print("df2 sumary: \n")
        print(average_time_success(df2))

    if test_accumulated_benchmark == True:
        for input_function, input_function_name, minimum in zip(
            input_functions, input_function_names, expected_minima
        ):
            df = run_smart_benchmark(
                our_smart_nelder_mead_method,
                input_function,
                minimum,
                [500],
                [0, 0],
                4,
                2,
                10,
                sample_size=25,
                number_of_candidates=3,
                algorithm_name="our nelder-mead-method",
                test_function_name=input_function_name,
            )
            print(df)
            plot_benchmark_results(df)
    pass
