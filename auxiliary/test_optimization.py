import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

from optimization_algorithms import nm_replace_final
from optimization_algorithms import nelder_mead_method
from optimization_algorithms import initial_simplex


FACTORS = list("cni")


def test_nm_replace_final():
    input_array = np.array([1, 2, 6, 4, 5, 3])
    expected_new_array = np.array([1, 2, 7, 4, 5, 3])

    indexes = np.argsort(input_array)
    x_new = 7
    new_array = nm_replace_final(input_array, indexes, x_new)
    assert_array_equal(new_array, expected_new_array)


def test_nelder_mead_method():
    input_function = lambda a: (a[0] + a[1]) ** 2 + 1
    expected_minimum = [0, 0]
    computed_minimum = nelder_mead_method(
        input_function, initial_simplex(2, [-10, 10]), 2
    )
    assert_array_almost_equal(computed_minimum, expected_minimum)
