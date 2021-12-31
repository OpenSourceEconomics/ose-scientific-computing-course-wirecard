import autograd.numpy as np
from autograd import grad, jacobian


# In this File we are going to implement the optimization algorithms we study in our assignment

# To get some practice with implementing optimization algorithms I implemented the 1 Dimensional Newton Method,
# and a naive optimization algorithms that uses the multidimensional newton method


def find_starting_point(f, domain, n, k=1):
    """Returns a candidate to start the local optimum finding process from.
    Args:
        f:              a function from \R^n to \R whose optimum we want to find
        domain:         the domain of the function in which we want to find the point (domain ist always a cube)
        n:              the dimension of the domain of the function
        k:              the amount of random points we draw to run the optimization on.




    Returns:
        out:            a candidate in domain^n to start the local search for an optimum from

    """
    # print("find_starting point bekommt n == ", n)
    A = np.random.rand(k, n)
    B = [f(domain[0] + (domain[1] - domain[0]) * x) for x in A]
    index = np.where(B == np.amin(B))
    x = A[index][0]
    y = domain[0] + (domain[1] - domain[0]) * x
    print("starting point is: ", y)
    return domain[0] + (domain[1] - domain[0]) * x


def stopping_criterion_x(inputs, values, input_tolerance, value_tolerance):
    eps = 0
    for input in inputs:
        eps = eps + abs(np.linalg.norm(input - inputs[0]))
    if eps < input_tolerance:
        return True
    else:
        return False


def stopping_criterion_y(inputs, values, input_tolerance, value_tolerance):
    eps = 0
    for value in values:
        eps = eps + abs(np.linalg.norm(value - values[0]))
    if eps < value_tolerance:
        return True
    else:
        return False


def stopping_criterion_y_or_x(inputs, values, input_tolerance, value_tolerance):
    if stopping_criterion_x(inputs, values, input_tolerance, value_tolerance):
        return True
    elif stopping_criterion_y(inputs, values, input_tolerance, value_tolerance):
        return True
    else:
        return False


def newton_method(f, df, x_n, eps=10 ** (-16), n=1000):
    """Return a candidate for a root of f, if the newton method starting at x_n converges.
    Args:
        f:              a function from \R^n to \R^n whose root we want to find
        df:             the jacobian of f; a function that takes x \in \R^n and returns a n*n matrix
        x_n:            a number within the domain of f from which to start the iteration
        eps:            sensitivity of of the root finding process
        n:              maximum of iterations before stopping the procedure
    Returns:
        out:            either an approximation for a root or a message if the procedure didnt converge
    """
    f_xn = f(x_n)
    while np.linalg.norm(f_xn) > eps and n > 0:
        sol = np.linalg.solve(df(x_n), -f_xn)
        x_n = x_n + sol
        f_xn = f(x_n)
        n = n - 1
    if np.linalg.norm(f_xn) < eps:
        return x_n
    else:
        return "Didnt converge."


def newton_method_new(
    f, df, x_n, x_tolerance=10 ^ (-6), y_tolerance=10 ^ (-6), computational_budget=1000
):
    """Return a candidate for a root of f, if the newton method starting at x_n converges.

    Args:
        f:              a function from \R^n to \R^n whose root we want to find
        df:             the jacobian of f; a function that takes x \in \R^n and returns a n*n matrix
        x_n:            a number within the domain of f from which to start the iteration
        eps:            sensitivity of of the root finding process
        n:              maximum of iterations before stopping the procedure



    Returns:
        out:            either an approximation for a root or a message if the procedure didnt converge

    """
    # Based on the inputs we decide which stopping criterion to take:
    if -np.inf < x_tolerance:
        if -np.inf < y_tolerance:
            stopping_criterion = lambda inputs, values: stopping_criterion_y_or_x(
                inputs, values, x_tolerance, y_tolerance
            )
        else:
            stopping_criterion = lambda inputs, values: stopping_criterion_x(
                inputs, values, x_tolerance, y_tolerance
            )
    elif -np.inf < y_tolerance:
        stopping_criterion = lambda inputs, values: stopping_criterion_y(
            inputs, values, x_tolerance, y_tolerance
        )
    else:
        assert (
            x_tolerance != y_tolerance
        ), "You should not set both x_tolerance and y_tolerance to -np.inf."

    # keeps track of iterations
    n = 0
    x = []
    val = []
    x.append(x_n)
    print("x equals: ", x)
    print("x[0] equals: ", x[0])
    new_value = f(x[0])
    val.append(new_value)
    n += 1  # increase n because f was called
    x.append(x[0] + np.linalg.solve(df(x[0]), -val[0]))
    val.append(f(x[1]))
    x = np.array(x)
    val = np.array(val)
    n += 1  # increase n because f was called

    # print("newton method bekommt als x_n: ", x_n)
    while stopping_criterion(x, val) == False and n < computational_budget:
        sol = np.linalg.solve(df(x_n), -val[1])
        n += 1  # increase n because df was called
        x[0] = x[1]
        x[1] = x[1] + sol
        # print("x_n equals to: ", x_n)
        # np.linalg.lstsq can deal with non invertibel matrices
        # x_n = x_n + np.linalg.solve(df(x_n), -f_xn)
        val[0] = val[1]
        val[1] = f(x[1])
        n += 1  # increase n because f was called

    return x[1], n


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
    f, dim, domain, eps_newton=10 ** (-6), eps_derivative=10 ** (-6), k=100, n=1000
):
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

    if len(domain) > 2:
        x_0 = domain
        # print("x_0 = domain; x_0 = ", x_0)
    elif len(domain) == 2:
        x_0 = np.array(find_starting_point(f, domain, dim)).astype(float)
        # print("x_0 by find_starting_point; x_0 = ", x_0)
    else:
        # print("domain ist: ",domain)
        print("domain ist nicht so wie sie sein sollte")
    # 2. compute derivative of f
    df = jacobian(f)
    # 3. compute jacobian of the derivative of f
    J = jacobian(df)
    # 4. run newton method on the derivative of f
    optimum = newton_method(df, J, x_0)
    # 5. return output of 4
    return optimum


def newton_based_naive_optimization(
    f, x_0, x_tolerance=10 ^ (-6), y_tolerance=10 ^ (-6), computational_budget=1000
):
    number_of_evaluations = 0

    df = jacobian(f)
    J = jacobian(df)

    optimum, number_of_evaluations = newton_method_new(
        df, J, x_0, x_tolerance, y_tolerance, computational_budget
    )

    return optimum, number_of_evaluations


if __name__ == "__main__":

    test_finding_starting_point = False
    test_initial_simplex = False

    if test_finding_starting_point:
        print(find_starting_point(lambda a: a[0] + a[1], [4, 6], 2))
