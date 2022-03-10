import pandas as pd
import math
import autograd.numpy as np
from autograd import grad, jacobian


from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


from auxiliary.nelder_mead_based_optimization_source import (
    initial_simplex,
    call_nelder_mead_method,
    nm_replace_final,
)
from auxiliary.newton_based_optimization_source import (
    naive_optimization,
)

from auxiliary.callable_algorithms import find_starting_points

from auxiliary.functions import rastrigin, griewank, levi_no_13, rosenbrock


FACTORS = list("cni")


def test_nm_replace_final():
    # creation of inputs
    input_array = np.array([1, 2, 6, 4, 5, 3])
    expected_new_array = np.array([1, 2, 7, 4, 5, 3])

    indexes = np.argsort(input_array)
    x_new = 7
    # computation by the function to be tested
    computed_array = nm_replace_final(input_array, indexes, x_new)
    # assert
    assert_array_equal(computed_array, expected_new_array)


def test_old_newton_opti():

    # initialization of inputs
    input_functions = [
        lambda a: 20 * (a[0] + a[1]) ** 2 + a[1] ** 4 + 1,
        lambda a: (1 / 200) * (a[0] + 1) ** 2 * (np.cos(a[1]) + 1) + a[1] ** 2,
        lambda a: (1 / 800) * (a[0] - 6) ** 4 * (np.sin(a[1]) + 3) + a[1] ** 4,
    ]

    input_functions_hard = [
        lambda a: griewank(a),
        lambda a: rosenbrock(a),
        lambda a: rastrigin(a),
    ]

    expected_minima = [[0, 0], [-1, 0], [6, 0]]

    expected_minima_hard = [[0, 0], [1, 1], [0, 0]]

    starting_points_hard = [[0.1, 0.1], [0.9, 0.8], [0.1, -0.1]]

    # names for readability of the error messages
    names_of_functions = [
        "lambda a: 20 * (a[0] + a[1]) ** 2 + a[1] ** 4 + 1",
        "lambda a: (1 / 200) * (a[0] + 1) ** 2 * (np.cos(a[1]) + 1) + a[1] ** 2",
        "lambda a : (1/800) * ( a[0] - 6) ** 4 * (np.sin(a[1])+ 3) + a[1] ** 4",
    ]

    names_of_functions_hard = ["griewank", "rosenbrock", "rastrigin"]

    # computation of minima
    computed_minima = [
        np.round(naive_optimization(input_function, [3, 3])[0])
        for input_function in input_functions
    ]

    computed_minima_hard = [
        np.round(
            naive_optimization(input_function, starting_point)[0]
        )  # [0] because naive optimization returns result and number of function calls
        for input_function, starting_point in zip(
            input_functions_hard, starting_points_hard
        )
    ]

    # assert
    for expected_minimum, computed_minimum, name in zip(
        expected_minima, computed_minima, names_of_functions
    ):

        assert_array_almost_equal(computed_minimum, expected_minimum, err_msg=name)

    for expected_minimum, computed_minimum, name in zip(
        expected_minima_hard, computed_minima_hard, names_of_functions_hard
    ):
        assert_array_almost_equal(computed_minimum, expected_minimum, err_msg=name)


def test_call_nelder_mead_method():

    # initialization of inputs
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

    starting_points_hard = [[0.1, 0.2], [0.9, 1.2], [0.2, -0.1]]

    # names of functions for readability of the error messages
    names_of_functions = [
        "lambda a: 20 * (a[0] + a[1]) ** 2 + a[1] ** 4 + 1",
        "lambda a: (1 / 200) * (a[0] + 1) ** 2 * (np.cos(a[1]) + 1) + a[1] ** 2",
        "lambda a : (1/800) * ( a[0] - 6) ** 4 * (np.sin(a[1])+ 3) + a[1] ** 4",
    ]
    names_of_functions_hard = ["griewank", "rosenbrock", "rastrigin"]

    # computation of minima
    computed_minima = [
        call_nelder_mead_method(input_function, initial_simplex(2, [-5, 5]))[0]
        for input_function in input_functions
    ]

    computed_minima_hard = [
        call_nelder_mead_method(
            input_function, initial_simplex(2, starting_point, 0.1)
        )[0]
        for input_function, starting_point in zip(
            input_functions_hard, starting_points_hard
        )
    ]

    # assert
    for expected_minimum, computed_minimum, name in zip(
        expected_minima, computed_minima, names_of_functions
    ):
        # print("expected minimum: ", expected_minimum, "computed minimum: ", computed_minimum)

        assert_array_almost_equal(computed_minimum, expected_minimum, err_msg=name)
        # print(index)

    for expected_minimum, computed_minimum, name in zip(
        expected_minima_hard, computed_minima_hard, names_of_functions_hard
    ):
        assert_array_almost_equal(computed_minimum, expected_minimum, err_msg=name)


if __name__ == "__main__":

    # to run tests manually (They do also pass when called by pytest)
    test_call_nelder_mead_method()
    test_old_newton_opti()

    print("All tests completed")
