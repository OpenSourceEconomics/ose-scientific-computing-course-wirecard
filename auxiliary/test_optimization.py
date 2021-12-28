import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


from optimization_algorithms_source import (
    nm_replace_final,
    nelder_mead_method,
    initial_simplex,
    nm_shrink,
    nelder_mead_method,
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


def test_nm_shrink():
    sigma = 0.5
    input_verts = [4, 6, 8, 2, 16, 10, 12, 14]
    expected_array = [3, 4, 5, 2, 9, 6, 7, 8]

    indexes = np.argsort(input_verts)
    computed_array = nm_shrink(input_verts, indexes, sigma)
    assert_array_equal(computed_array, expected_array)


def test_nelder_mead_method():
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

    names_of_functions = [
        "lambda a: 20 * (a[0] + a[1]) ** 2 + a[1] ** 4 + 1",
        "lambda a: (1 / 200) * (a[0] + 1) ** 2 * (np.cos(a[1]) + 1) + a[1] ** 2",
        "lambda a : (1/800) * ( a[0] - 6) ** 4 * (np.sin(a[1])+ 3) + a[1] ** 4",
    ]
    names_of_functions_hard = ["griewank", "rosenbrock", "rastrigin"]

    computed_minima = [
        nelder_mead_method(input_function, initial_simplex(2, [-5, 5]), 2)
        for input_function in input_functions
    ]
    for expected_minimum, computed_minimum, name in zip(
        expected_minima, computed_minima, names_of_functions
    ):
        # print("expected minimum: ", expected_minimum, "computed minimum: ", computed_minimum)

        assert_array_almost_equal(computed_minimum, expected_minimum, err_msg=name)
        # print(index)


if __name__ == "__main__":
    test_nelder_mead_method()
