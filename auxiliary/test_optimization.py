import pandas as pd
import math
import autograd.numpy as np
from autograd import grad, jacobian
from aux_m.test_problems import problem_application_correct_1


from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


from nelder_mead_based_optimization_source import (
    initial_simplex,
    call_nelder_mead_method,
    nm_replace_final,
)
from newton_based_optimization_source import (
    naive_optimization,
)

from functions import rastrigin, griewank, levi_no_13, rosenbrock


FACTORS = list("cni")


def test_nm_replace_final():
    input_array = np.array([1, 2, 6, 4, 5, 3])
    expected_new_array = np.array([1, 2, 7, 4, 5, 3])

    indexes = np.argsort(input_array)
    x_new = 7
    computed_array = nm_replace_final(input_array, indexes, x_new)
    assert_array_equal(computed_array, expected_new_array)


def test_functions_used_here():
    input_functions_raw = [
        lambda a: 20 * (a[0] + a[1]) ** 2 + a[1] ** 4 + 1,
        lambda a: (1 / 200) * (a[0] + 1) ** 2 * (math.cos(a[1]) + 1) + a[1] ** 2,
        lambda a: (1 / 800) * (a[0] - 6) ** 4 * (math.sin(a[1]) + 3) + a[1] ** 4,
    ]

    f_1 = lambda b: input_functions_raw[0](b)
    f_2 = lambda b: input_functions_raw[1](b)
    f_3 = lambda b: input_functions_raw[2](b)

    input_functions = [lambda b: f_1(b), lambda b: f_2(b), lambda b: f_3(b)]

    input_functions_hard = [
        lambda a: griewank(a),
        lambda a: rosenbrock(a),
        lambda a: rastrigin(a),
    ]

    expected_minima = [[0, 0], [-1, 0], [6, 0]]

    expected_minima_hard = [[0, 0], [1, 1], [0, 0]]

    names_of_functions = [
        "lambda a: 20 * (a[0] + a[1]) ** 2 + a[1] ** 4 + 1",
        "lambda a: (1 / 200) * (a[0] + 1) ** 2 * (np.cos(a[1]) + 1) + a[1] ** 2",
        "lambda a : (1/800) * ( a[0] - 6) ** 4 * (np.sin(a[1])+ 3) + a[1] ** 4",
    ]
    names_of_functions_hard = ["griewank", "rosenbrock", "rastrigin"]

    A = []
    for function, name in zip(input_functions, names_of_functions):
        print(name)
        A.append(function([1, 2]))
        A.append(function([2, 3]))
        A.append(function([4, 2]))
        # A.append(function(np.array([1,2])))
        # A.append(function(np.array([2,3])))
        # A.append(function(np.array([4,2])))
        df = jacobian(function)
        J = jacobian(function)
        print(df(np.array([1.0, 2.0])))
        # A.append(J([1,2]))


def test_old_newton_opti():
    input_functions = [
        lambda a: 20 * (a[0] + a[1]) ** 2 + a[1] ** 4 + 1,
        lambda a: (1 / 200) * (a[0] + 1) ** 2 * (np.cos(a[1]) + 1) + a[1] ** 2,
        lambda a: (1 / 800) * (a[0] - 6) ** 4 * (np.sin(a[1]) + 3) + a[1] ** 4,
    ]

    input_functions_hard = [
        lambda a: griewank(a),
        lambda a: rosenbrock(a),
        lambda a: rastrigin(a),
        lambda a: levi_no_13(a),
    ]

    expected_minima = [[0, 0], [-1, 0], [6, 0]]

    expected_minima_hard = [[0, 0], [1, 1], [0, 0], [1, 1]]

    names_of_functions = [
        "lambda a: 20 * (a[0] + a[1]) ** 2 + a[1] ** 4 + 1",
        "lambda a: (1 / 200) * (a[0] + 1) ** 2 * (np.cos(a[1]) + 1) + a[1] ** 2",
        "lambda a : (1/800) * ( a[0] - 6) ** 4 * (np.sin(a[1])+ 3) + a[1] ** 4",
    ]

    names_of_functions_hard = ["griewank", "rosenbrock", "rastrigin", "levi_no_13"]

    computed_minima = [
        np.round(
            naive_optimization(input_function, [-3, 3])[0]
        )  # [0] because naive optimization returns result and number of function calls
        for input_function in input_functions_hard
    ]

    print("computations worked")
    for expected_minimum, computed_minimum, name in zip(
        expected_minima, computed_minima, names_of_functions_hard
    ):
        # print("expected minimum: ", expected_minimum, "computed minimum: ", computed_minimum)

        assert_array_almost_equal(computed_minimum, expected_minima_hard, err_msg=name)


def test_call_nelder_mead_method():
    input_functions = [
        lambda a: 20 * (a[0] + a[1]) ** 2 + a[1] ** 4 + 1,
        lambda a: (1 / 200) * (a[0] + 1) ** 2 * (math.cos(a[1]) + 1) + a[1] ** 2,
        lambda a: (1 / 800) * (a[0] - 6) ** 4 * (math.sin(a[1]) + 3) + a[1] ** 4,
    ]

    input_functions_hard = [
        lambda a: griewank(a),
        lambda a: rosenbrock(a),
        lambda a: rastrigin(a),
    ]

    expected_minima = [[0, 0], [-1, 0], [6, 0]]

    expected_minima_hard = [[0, 0], [1, 1], [0, 0]]

    names_of_functions = [
        "lambda a: 20 * (a[0] + a[1]) ** 2 + a[1] ** 4 + 1",
        "lambda a: (1 / 200) * (a[0] + 1) ** 2 * (np.cos(a[1]) + 1) + a[1] ** 2",
        "lambda a : (1/800) * ( a[0] - 6) ** 4 * (np.sin(a[1])+ 3) + a[1] ** 4",
    ]
    names_of_functions_hard = ["griewank", "rosenbrock", "rastrigin"]

    print(call_nelder_mead_method(input_functions[0], initial_simplex(2, [-5, 5]), 2))
    computed_minima = [
        call_nelder_mead_method(input_function, initial_simplex(2, [-5, 5]), 2)[0]
        for input_function in input_functions
    ]
    for expected_minimum, computed_minimum, name in zip(
        expected_minima, computed_minima, names_of_functions
    ):
        # print("expected minimum: ", expected_minimum, "computed minimum: ", computed_minimum)

        assert_array_almost_equal(computed_minimum, expected_minimum, err_msg=name)
        # print(index)


def problem_application(x):

    #### define problem of the application 3 dimensional case
    B = 1
    lambda_param = 1
    alpha_1 = 0.0427
    alpha_2 = 0.0015
    alpha_3 = 0.0285
    sigma_x_1 = 0.01  ### square std deviation to get variance
    sigma_x_2 = 0.01089936
    sigma_x_3 = 0.01990921
    sigma_x_12 = 0.0018
    sigma_x_13 = 0.0011
    sigma_x_23 = 0.0026

    # exp_returns=-alpha_1*x[0]-alpha_2*x[1]-alpha_3*x[2]

    disutility_volatility = (
        (sigma_x_1 * (x[0] ** 2))
        + (x[1] ** 2) * sigma_x_2
        + (x[2] ** 2) * sigma_x_3
        + 2 * sigma_x_12 * x[0] * x[1]
        + 2 * sigma_x_13 * x[0] * x[2]
        + 2 * x[1] * x[2] * sigma_x_23
    )

    #### disutility is coded without lambda
    return disutility_volatility


if __name__ == "__main__":
    # test_functions_used_here()
    # test_nelder_mead_method()
    test_call_nelder_mead_method()
    verts = [[1, 1, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0]]
    result = call_nelder_mead_method(problem_application, verts, 3)
    print(result)
    # test_old_newton_opti()

    print("All tests completed")
