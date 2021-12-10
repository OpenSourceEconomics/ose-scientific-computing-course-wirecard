import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

from optimization_algorithms import nm_replace_final
from optimization_algorithms import nelder_mead_method
from optimization_algorithms import initial_simplex
from optimization_algorithms import nm_shrink


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
    input_function = lambda a: 20 * (a[0] + a[1]) ** 2 + 1
    expected_minimum = [0, 0]
    computed_minimum = nelder_mead_method(
        input_function, initial_simplex(2, [-1, 1]), 2
    )
    assert_array_almost_equal(computed_minimum, expected_minimum)
