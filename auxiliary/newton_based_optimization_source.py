from itertools import count
import autograd.numpy as np
from autograd import grad, jacobian


# In this File we implented a naive local optimization algorithm that uses the newton
# method and requires the funciton of interest to be twice continuously differentiable

# this function is not used yet but could be integrated later to give the optimization algorithms more flexibility
def pick_stopp_criterion(x_tolerance, y_tolerance):
    """Return the correct termination criterion.

    Args:
        x_tolerance:    a real valued number or -infty
        y_tolerance:    a real valued number or -infty


    Returns:
        stopping_criterion: The stopping-criterion we want to use in the recursion of the nelder-mead_algorithm

    """
    if -np.inf < x_tolerance:
        if -np.inf < y_tolerance:
            stopping_criterion = lambda inputs, values: stopping_criterion_x_or_y(
                inputs, values, x_tolerance, y_tolerance
            )
            return stopping_criterion
        else:
            stopping_criterion = lambda inputs, values: stopping_criterion_x(
                inputs, values, x_tolerance, y_tolerance
            )
            return stopping_criterion
    elif -np.inf < y_tolerance:
        stopping_criterion = lambda inputs, values: stopping_criterion_y(
            inputs, values, x_tolerance, y_tolerance
        )
        return stopping_criterion
    else:
        assert (
            x_tolerance != y_tolerance
        ), "You should not set both x_tolerance and y_tolerance to -np.inf."


# checks wether stopping criterion in x is reached
def stopping_criterion_x(x, y, x_tolerance, y_tolerance):
    """Return False if x < x_tolerance, else True.

    Args:
        x:          The current x value
        y:          the current y value
        x_tolerance:    a real valued number or -infty
        y_tolerance:    a real valued number or -infty

    Returns:
        out: Boolean

    """
    if abs(np.linalg.norm(x)) < x_tolerance:
        return False
    else:
        return True


# checks wether stopping criterion in y is reached
def stopping_criterion_y(x, y, x_tolerance, y_tolerance):
    """Return True if y < y_tolerance, else false.

    Args:
        x:          The current x value
        y:          the current y value
        x_tolerance:    a real valued number or -infty
        y_tolerance:    a real valued number or -infty

    Returns:
        out: Boolean

    """
    norm = abs(np.linalg.norm(y))
    if norm < y_tolerance:
        # print("return false")
        return False
    else:
        return True


# checks wether either the stopping criterion in y or in x is reached
def stopping_criterion_x_or_y(x, y, x_tolerance, y_tolerance):
    """Return True if both terminate_criterion_y and terminate_criterion_x are True, else false.

    Args:
        verts:          The verticies of a simplex
        f_difference:   The absolute difference of the last and secondlast function_value
        x_tolerance:    a real valued number or -infty
        y_tolerance:    a real valued number or -infty

    Returns:
        out: Boolean

    """
    if stopping_criterion_x(x, y, x_tolerance, y_tolerance):
        if stopping_criterion_y(x, y, x_tolerance, y_tolerance):
            return True
        else:
            return False
    else:
        return False


# the multi-dimensional newton method
def newton_method(
    f,
    df,
    x_n,
    x_tolerance=1e-6,
    y_tolerance=1e-6,
    n=1000,
    stopping_criterium=stopping_criterion_y,
):
    """Return a candidate for a root of f, if the newton method starting at x_n converges.
    Args:
        f:                   a function from \R^n to \R^n whose root we want to find
        df:                  the jacobian of f; a function that takes x \in \R^n and returns a n*n matrix
        x_n:                 a number within the domain of f from which to start the iteration
        x_tolerance:         a postitive real number
        y_tolerance:         a positive real number
        n:                   maximum of iterations before stopping the procedure
        stopping_criterim:   a function that returns a boolean
    Returns:
        out_1:          candidate for the root
        out_2:          number of function evaluations
    """

    # default stopping criterion for now y
    # this can still be adapted later to make the algorithms more flexible

    # initilize variables:
    calls_of_f_or_df = 0
    f_xn = f(x_n)
    calls_of_f_or_df = calls_of_f_or_df + 1
    n = n - 1
    # iteration of the newton method
    while stopping_criterium(x_n, f_xn, x_tolerance, y_tolerance) and n > 0:

        # solve linear equations
        sol = np.linalg.solve(df(x_n), -f_xn)

        # adjust count of function evaluations
        calls_of_f_or_df = calls_of_f_or_df + 1
        n = n - 1

        x_n = x_n + sol
        f_xn = f(x_n)

        # adjust count of function evaluations
        calls_of_f_or_df = calls_of_f_or_df + 1
        n = n - 1
    return x_n, calls_of_f_or_df


# This function is not used anymore and can be deleted 10.03.2022 Daniel
def new_naive_optimization(
    f, starting_point, x_tolerance=1e-6, y_tolerance=1e-6, computational_budget=100
):
    x_0 = np.array(starting_point).astype(float)
    df = jacobian(f)
    J = jacobian(jacobian(f))
    stopping_criterion = pick_stopp_criterion(x_tolerance, y_tolerance)
    pass


# a local optimization search based on the newton method
def naive_optimization(
    f, starting_point, x_tolerance=1e-6, y_tolerance=1e-6, computational_budget=100
):
    """Return a candidate for an optimum of f, if the procedure converges.

    Args:
        f:                  a function from \R^n to \R whose optimum we want to find
        starting_point:     the starting point of the iteration
        x_tolerance:        a positive real number
        y_tolerance:        a positive real number
        computation_budget: the number of maximal function evaluations

    Returns:
        out: a candidate for a local optimum and the number of function evaluations

    """
    # 1. Set the starting point of the iteration
    x_0 = np.array(starting_point).astype(float)

    # 2. compute derivative of f
    df = jacobian(f)
    # 3. compute jacobian of the derivative of f
    J = jacobian(df)
    # 4. choose stopping criterion, this is not integrated yet
    stopping_criterion = pick_stopp_criterion(x_tolerance, y_tolerance)
    # 5. run newton method on the derivative of f
    optimum, calls = newton_method(
        df, J, x_0, x_tolerance, y_tolerance, computational_budget
    )
    # 6. return output of 4

    return np.round(optimum), calls
