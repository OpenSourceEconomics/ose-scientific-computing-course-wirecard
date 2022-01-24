from itertools import count
import autograd.numpy as np
from autograd import grad, jacobian


# In this File we are going to implement the optimization algorithms we study in our assignment

# To get some practice with implementing optimization algorithms I implemented the 1 Dimensional Newton Method,
# and a naive optimization algorithms that uses the multidimensional newton method


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


def find_starting_point(f, domain, n, k=10000):
    """Returns a candidate to start the local optimum finding process from.
    Args:
        f:              a function from \R^n to \R whose optimum we want to find
        domain:         the domain of the function in which we want to find the point (domain ist always a cube)
        n:              the dimension of the domain of the function
        k:              the amount of random points we draw to run the optimization on.

    Returns:
        out:            a candidate in domain^n to start the local search for an optimum from

    """
    A = np.random.rand(k, n)
    B = [f(domain[0] + (domain[1] - domain[0]) * x) for x in A]
    index = np.where(B == np.amin(B))
    x = A[index][0]
    y = domain[0] + (domain[1] - domain[0]) * x
    return domain[0] + (domain[1] - domain[0]) * x


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

    calls_of_f_or_df = 0
    f_xn = f(x_n)
    calls_of_f_or_df = calls_of_f_or_df + 1
    n = n - 1
    while stopping_criterium(x_n, f_xn, x_tolerance, y_tolerance) and n > 0:

        sol = np.linalg.solve(df(x_n), -f_xn)

        calls_of_f_or_df = calls_of_f_or_df + 1
        n = n - 1

        x_n = x_n + sol
        f_xn = f(x_n)
        calls_of_f_or_df = calls_of_f_or_df + 1
        n = n - 1
    return x_n, calls_of_f_or_df


# TODO - write a newto-method routine for michael that takes f, startpoint x_0
#      - computational budget/ maximale anzahl an iterationen als argument
#      - stop criterium für terminate als stopp kriterium
#      - return die minimalstelle der function
#      - return die anzahl der functionsevaluationen


#      Wenn stopping_tolerance_x = -inf dann nimm bitte das andere
#       wennn keines = -inf ist das was als erstes eintrit
#       wenn beide -inf sind -> fehler

# def minimization(f,starting_point, stopping_tolerance_xwert, stopping_tolerance_functionswert, computational _budget):
# return optimum, anzahl_evaluationen


# TODO the same for the naive-optimization


# TODO

# Maybe we could implement the neewton-method as a class such that we can safe, call and change the values alpha, gamma, rho and sigma


# The following naive optimization is a bit messy right now we are working with: newton_based_naive_optimization
def naive_optimization(
    f, starting_point, x_tolerance=1e-6, y_tolerance=1e-6, computational_budget=100
):
    # print("naive y tol: ", y_tolerance)
    """Return a candidate for an optimum of f, if the procedure converges.

    Args:
        f:              a function from \R^n to \R whose optimum we want to find
        dim:            dimension of the function
        domain:         an array A = [a,b] that defines the domain of the function (always a square) or None
        eps_newton:     sensitivity of the root finding process
        eps_derivative: sensitivity of the derivative approximation
        k:              number of gridpoints in each axis
        n:              maximum of iterations before stopping the procedure



    Returns:
        out: either an approximation for an optimum or a message if the procedure didnt converge

    """
    # 1. find point x_0 to start iteration from
    # For now we treat domain as the starting point of the iteration

    x_0 = np.array(starting_point).astype(float)

    """
    if len(domain) > 2:
        x_0 = domain
        # print("x_0 = domain; x_0 = ", x_0)
    elif len(domain) == 2:
        x_0 = np.array(find_starting_point(f, domain, dim)).astype(float)
        # print("x_0 by find_starting_point; x_0 = ", x_0)
    else:
        # print("domain ist: ",domain)
        print("domain ist nicht so wie sie sein sollte")
    """

    # 2. compute derivative of f
    df = jacobian(f)
    # 3. compute jacobian of the derivative of f
    J = jacobian(df)
    # 4. run newton method on the derivative of f
    stopping_criterion = stopping_criterion_y
    optimum, calls = newton_method(
        df, J, x_0, x_tolerance, y_tolerance, computational_budget
    )

    print(calls)
    # 5. return output of 4
    return optimum, calls


if __name__ == "__main__":

    test_finding_starting_point = False
    test_initial_simplex = False

    if test_finding_starting_point:
        print(find_starting_point(lambda a: a[0] + a[1], [4, 6], 2))
