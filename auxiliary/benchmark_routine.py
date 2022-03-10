import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns


from auxiliary.callable_algorithms import (
    our_simple_nelder_mead_method,
    our_simple_newton_based_optimization,
    global_nelder_mead_method,
    global_newton_based_optimization,
    global_optimization_BOBYQA,
)
from auxiliary.functions import rastrigin, griewank, rosenbrock, levi_no_13

# In this file we implement the benchmark routine that we use to benchmark the nelder-mead-method
# we implemented and compare its performace to algorithms from the established libary NLOPT

# This file includes to benchmarks
# the benchmark 'benchmark_simple_optimization is used for algorithms that start from a signle point
# the benchmark 'benchmark_smart_optimization' is used for algorithms that get a domain and find their own starting points

# the function with the prefix 'run_' simply execute the corresponding benchmark for multiple computational budgets and safe that data in a dataframe

# This function creates starting points fot the 'benchmark_simple_optimization'
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
    # get random numbers
    A = np.random.rand(n, dimension) * domain_radius
    # ensure they are in the domain
    A = [domain_center + x for x in A]
    return A


# this function computes the measures of interests after the benchmarks are run
def average_time_success(df):
    """Return a dataframe which contains the results of the optimization algorithm applied to the
       test-funcion f with n different starting point.

    Args:
        df:    non-empty dataframe with the collumns: algorithm, test_function,computational_budget, smaple_size,success, time function_evaluations




    Returns:
        out:              datafram with one entry and the same collumns

    """
    # create new dataframe
    df_new = pd.DataFrame([])
    # safe the name of the algorithm
    df_new["algorithm"] = pd.Series([df.iloc[0]["algorithm"]])
    # safe the name of the test-function
    df_new["test_function"] = pd.Series([df.iloc[0]["test_function"]])
    # safe the computational_budget used
    df_new["computational_budget"] = pd.Series([df.iloc[0]["computational_budget"]])
    # safe the sample size
    df_new["sample_size"] = pd.Series([df.iloc[0]["sample_size"]])
    # compute and safe the success rate
    df_new["success_rate"] = pd.Series([df["success"].mean()])
    # compute and safe the average time per optimization
    df_new["average_time"] = pd.Series([df["time"].mean()])
    # compute and safe the average number of function evaluations taken
    df_new["average_function_evaluations"] = pd.Series(
        [df["function_evaluations"].mean()]
    )
    return df_new


# This function performs the benchmark for algorithms that take the domain as input and find their own starting points
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

    """Return a dataframe which contains the results of the optimization algorithm applied number_of_optimizations many times to the
       test-funcion f.

    Args:
        f:                       a function from \R^n to \R whose optimum we want to find
        algorithm:               the optimization algorithm we use for the optimization
        computation_budget:      the number of maximal function evaluations
        domain_center:           a point in \R^n
        domain_radius:           a positive real number
        number_of_optimizations: a natural number
        dimension:               the dimension of the problem
        optimum:                 the correct optimum of f
        algo_name:               the name of the algorithm as a string
        t_func_name:             the name of the test function
        x_tolerance:             a positive real number
        y_tolerance:             a positive real number
        sample_size:             The optimization-intern number of points that get considered as starting points for the local search
        number_of_candidates:    The optimization-intern number of  starting points from which we start the local search


    Returns:
        out:               A dataframe that contains the results and the number of function evaluation for
                           each attemp to optimize f

    """
    # create dataframe
    df = pd.DataFrame([], columns=["computed_result", "function_evaluations", "time"])

    # run the optimizations
    for i in range(number_of_optimizations):
        # start timer to take the time
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
        # end timer
        end = time.time()
        a = list(a)
        # compute time taken and add as new collumn
        a = pd.Series(
            a + [end - start],
            index=df.columns,
        )
        df = df.append(a, ignore_index=True)

    # compute further measures we might want to analyse later
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


# benchmark an algorithm that starts from a single point
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
    # get starting points
    starting_points = get_starting_points(n, domain_center, domain_radius, dimension)

    # create dataframe
    df = pd.DataFrame([], columns=["computed_result", "function_evaluations", "time"])

    # run optimizations from starting points
    for starting_point in starting_points:
        # start timer
        start = time.time()
        a = algo(f, starting_point, x_tolerance, y_tolerance, computational_budget)
        # end timer
        end = time.time()
        a = list(a)
        # compute time passed and add as collumn to the df
        a = pd.Series(
            a + [end - start],
            index=df.columns,
        )
        df = df.append(a, ignore_index=True)

    # compute further measures we might want to analyse later
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


# run the benchmark for algorithms that start from a single point for multiple computational budgets
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
    """Return a dataframe which contains the results of the optimization algorithm applied to the
       test-funcion f with n different starting point for different computational budgets.

    Args:
        algorithm:                      The optimization algorithm we use for the optimization
        test_function:                  a function from \R^n to \R whose optimum we want to find
        optimum:                        The correct optimum of f
        computational_budgets.          an array of computational budgets
        domain_center:                  The center of the domain in which we want to start the optimization
        domain_radius:                  The radius of the domain
        dimension:                      The dimension of the problem f
        n:                              The number or starting points from which we start the optimization
        algorithm_name="unknown":       the name of the algorithm as a string
        test_function_name="unknown":   the name of the testfunction as a string
        x_tolerance=1e-6:                a positive real number
        y_tolerance=1e-6:                a positive real number



    Returns:
        out:               A dataframe that contains the averaged results and the number of function evaluation for
                           each attemp to optimize f

    """
    # initilize variables

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
    # for the user to have a rough idea how long the benchmark will still take
    count = 1
    print("Benchmark ", count, " out of ", len(computational_budgets), "done.")
    # run benchmark for all computational budgets
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


# run the benchmark for algorithms find their own starting points for multiple computational budgets
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
    """Return a dataframe which contains the averaged results of the optimization algorithm applied number_of_optimizations many times to the
       test-funcion f for all computational budgets

    Args:
        algorithm:               the optimization algorithm we use for the optimization
        f:                       a function from \R^n to \R whose optimum we want to find
        optimum:                 the correct optimum of f
        computation_budget:      an array of computational budgets
        domain_center:           a point in \R^n
        domain_radius:           a positive real number
        dimension:               the dimension of the problem
        number_of_optimizations: a natural number
        sample_size:             The optimization-intern number of points that get considered as starting points for the local search
        number_of_candidates:    The optimization-intern number of  starting points from which we start the local search
        algorithm_name:          the name of the algorithm as a string
        test_function_name:      the name of the test function as a string
        x_tolerance:             a positive real number
        y_tolerance:             a positive real number


    Returns:
        out:               A dataframe that contains the averaged results and the number of function evaluation for
                           each attemp to optimize f, for each computational budget

    """
    # initilize variables

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
    # for the user to have a rough idea how long the benchmarks still takes
    count = 1
    print("Benchmark ", count, " out of ", len(computational_budgets), "done.")

    # run benchmarks for remaining computational budgets
    for computational_budget in computational_budgets[1:]:
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


# this function plots the benchmark results for different algorithms but a single test_functions
def plot_benchmark_results(df):
    """Plot the success-rate given the computational budget of benchmark whose results are saved in df.

    Args:
        df:               A dataframe with the collumns: computational_budget, success_rate and algorithm


    Returns:
        out:              None

    """
    # plot the data
    sns.lineplot(x="computational_budget", y="success_rate", hue="algorithm", data=df)
    plt.show()


if __name__ == "__main__":

    test_benchmark_optimization = False
    test_accumulated_benchmark = False

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

    input_functions_hard = [
        lambda a: griewank(a),
        lambda a: rosenbrock(a),
        lambda a: rastrigin(a),
    ]

    expected_minima_hard = [[0, 0], [1, 1], [0, 0]]

    names_of_functions_hard = ["griewank", "rosenbrock", "rastrigin"]

    df = run_smart_benchmark(
        global_optimization_BOBYQA,
        griewank,
        [0, 0],
        [1000, 2000, 3000],
        [0, 0],
        100,
        2,
        10,
        algorithm_name="BOBYQA",
        test_function_name="griewank",
    )
    df = pd.concat(
        [
            df,
            run_smart_benchmark(
                global_optimization_BOBYQA,
                rosenbrock,
                [1, 1],
                [1000, 2000, 3000],
                [0, 0],
                100,
                2,
                10,
                algorithm_name="BOBYQA",
                test_function_name="rosenbrock",
            ),
        ]
    )
    df = pd.concat(
        [
            df,
            run_smart_benchmark(
                global_optimization_BOBYQA,
                rastrigin,
                [0, 0],
                [1000, 2000, 3000],
                [0, 0],
                100,
                2,
                10,
                algorithm_name="BOBYQA",
                test_function_name="rastrigin",
            ),
        ]
    )

    fig, axes = plt.subplots(ncols=1, nrows=3)

    for function, ax in zip(names_of_functions_hard, axes.flat):
        sns.lineplot(
            ax=ax,
            x="computational_budget",
            y="success_rate",
            hue="algorithm",
            data=df.loc[df["test_function"] == function],
        )
    plt.show()


"""

    df0 = run_smart_benchmark(
        optimization_smart_BOBYQA_NLOPT,
        griewank,
        [0, 0],
        [
            500,
            1000,
            1500,
            2000,
            2500,
            3000,
            3500,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
        ],
        [0, 0],
        100,
        2,
        10,
        algorithm_name="BOBYQA",
    )
    df1 = run_smart_benchmark(
        our_smart_nelder_mead_method,
        griewank,
        [0, 0],
        [
            500,
            1000,
            1500,
            2000,
            2500,
            3000,
            3500,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
        ],
        [0, 0],
        100,
        2,
        10,
        algorithm_name="Nelder-Mead",
    )

    df0 = pd.concat([df0, df1])

    plot_benchmark_results(df0)

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
"""
