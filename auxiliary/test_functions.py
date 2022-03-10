import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


from functions import (
    rastrigin_instance,
    griewank_instance,
    levi_no_13_instance,
    rosenbrock_instance,
)


# This File includes tests to check wether we implemented the test-functions correctly.
# All test can be executed with pytask

# test for the rastrigin function
def test_rastrigin():
    # get inputs
    inputs = create_inputs()
    # get function and define expected outs
    function = rastrigin_instance(3)
    expected_outs = np.array([15, 111, 139, 31, 2, 46, 1180, 3])
    # compute outs and assert
    computed_outs = np.array([function.value(x) for x in inputs])
    assert_array_almost_equal(expected_outs, computed_outs)


# test for the griewank function
def test_griewank():
    # get inputs
    inputs = create_inputs()
    # get function and define expected outs
    function = griewank_instance(3)
    expected_outs = np.array([2.084, 2.470, 2.774, 2.113, 1.245, 2.608, 7.256, 1.599])
    # compute outs and assert
    computed_outs = np.round([function.value(x) for x in inputs], 3)
    assert_array_almost_equal(expected_outs, computed_outs)


# test for the levi o. 13 function
def test_levi_no_13():
    # get inputs
    inputs = create_inputs()
    # get function and define expected outs
    function = levi_no_13_instance(3)
    expected_outs = np.array([6, 78, 134, 22, 3, 63, 1065, 6])
    # compute outs and assert
    computed_outs = np.array([function.value(x) for x in inputs])
    assert_array_almost_equal(expected_outs, computed_outs)


# test for the rosenbrock function
def test_rosenbrock():
    # get inputs
    inputs = create_inputs()
    # get function and define expected outs
    function = rosenbrock_instance(3)
    expected_outs = np.array([202, 120242, 380598, 55418, 202, 50527, 18909781, 505])
    # compute outs and assert
    computed_outs = np.array([function.value(x) for x in inputs])
    assert_array_almost_equal(expected_outs, computed_outs)


def create_inputs():
    out = np.array(
        [
            [1, 2, 3],
            [5, 6, 7],
            [-8, 5, 7],
            [5, 2, -1],
            [0, 1, 0],
            [2, -4, -5],
            [19, 17, 23],
            [1, -1, 0],
        ]
    )
    return out


if __name__ == "__main__":
    test_rastrigin()
    test_rosenbrock()
    test_griewank()
    test_levi_no_13()
    print("All test completed.")
