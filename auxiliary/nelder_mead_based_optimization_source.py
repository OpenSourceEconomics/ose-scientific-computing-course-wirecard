import numpy as np


def initial_simplex(dim, domain, eps=0.5):
    """Return a dim- dimensional simplex within the cube domain^n
    Args:
        dim:           the dimension we are working with
        domain:        edges of the domain

    Returns:
        A:           the verticies of the simplex in an dim+1 dimensional array

    """
    A = np.random.rand(dim + 1, dim) * eps
    A = [domain[0] + x * (domain[1] - domain[0]) for x in A]

    return A


def pick_terminate_criterion(x_tolerance, y_tolerance):
    """Return the correct termination criterion.

    Args:
        x_tolerance:    a real valued number or -infty
        y_tolerance:    a real valued number or -infty


    Returns:
        stopping_criterion: The stopping-criterion we want to use in the recursion of the nelder-mead_algorithm

    """
    if -np.inf < x_tolerance:
        if -np.inf < y_tolerance:
            stopping_criterion = lambda inputs, values: terminate_criterion_x_or_y(
                inputs, values, x_tolerance, y_tolerance
            )
            return stopping_criterion
        else:
            stopping_criterion = lambda inputs, values: terminate_criterion_x(
                inputs, values, x_tolerance, y_tolerance
            )
            return stopping_criterion
    elif -np.inf < y_tolerance:
        stopping_criterion = lambda inputs, values: terminate_criterion_y(
            inputs, values, x_tolerance, y_tolerance
        )
        return stopping_criterion
    else:
        assert (
            x_tolerance != y_tolerance
        ), "You should not set both x_tolerance and y_tolerance to -np.inf."


def terminate_criterion_x_or_y(verts, f_difference, x_tolerance, y_tolerance):
    """Return True if both terminate_criterion_y and terminate_criterion_x are True, else false.

    Args:
        verts:          The verticies of a simplex
        f_difference:   The absolute difference of the last and secondlast function_value
        x_tolerance:    a real valued number or -infty
        y_tolerance:    a real valued number or -infty

    Returns:
        out: Boolean

    """
    if terminate_criterion_x(verts, f_difference, x_tolerance, y_tolerance):
        if terminate_criterion_y(verts, f_difference, x_tolerance, y_tolerance):
            return True
        else:
            return False
    else:
        return False


def terminate_criterion_x(verts, f_difference, x_tolerance, y_tolerance):
    """Return True if Var(verts) < x_tolerance, else false.

    Args:
        verts:          The verticies of a simplex
        f_difference:   The absolute difference of the last and secondlast function_value
        x_tolerance:    a real valued number or -infty
        y_tolerance:    a real valued number or -infty

    Returns:
        out: Boolean

    """
    eps = 0
    for vert in verts:
        eps += abs(np.linalg.norm(vert - verts[0]))
    eps = eps / len(verts)
    # print("Summed distance = ", eps)
    if eps < x_tolerance:
        return True
    if eps > 50:
        return True
    else:
        return False


def terminate_criterion_y(verts, f_difference, x_tolerance, y_tolerance):
    """Return True if f_difference < y_tolerance, else false.

    Args:
        verts:          The verticies of a simplex
        f_difference:   The absolute difference of the last and secondlast function_value
        x_tolerance:    a real valued number or -infty
        y_tolerance:    a real valued number or -infty

    Returns:
        out: Boolean

    """
    if f_difference < y_tolerance:
        return True
    else:
        return False


def nm_replace_final(verts, indexes, x_new):
    """Replaces the verticy with the highest associated function value with x_new.
    Args:
        verts:      Array of verticies
        indexes:    Arrays if intergers ordering verts according to assiciated function values
        x_new:      New verticy  to be added to verts

    Returns:
        out:            a candidate in domain^n to start the local search for an optimum from

    """
    new_verts = []
    for i in range(len(verts)):
        new_verts.append(verts[i])
    new_verts[indexes[-1]] = x_new
    new_verts = np.array(new_verts)
    return new_verts


# TODO

# Maybe we could implement the nelder-mead-method as a class such that we can safe, call and change the values alpha, gamma, rho and sigma


def call_nelder_mead_method(
    f,
    verts,
    dim,
    f_difference=10,
    calls=0,
    terminate_criterion=terminate_criterion_x,
    x_tolerance=1e-6,
    y_tolerance=1e-6,
    computational_budget=1000,
    alpha=1,
    gamma=2,
    rho=0.5,
    sigma=0.5,
    values=[],
):
    """Return an approximation of a local optimum.

    Args:
        f:                                  a real valued n_dimensional function
        verts:                              an array with n+1 n-dimensional vectors
        dim:                                a integer (equal to n)
        f_difference:                       the difference between the last and second last best approximation
        calls:                              the number of evaluations of f so far
        terminate_criterion:                the termination criterion we are using (a function that returns a boolean)
        x_tolerance:                        A positive real number
        y_tolerance:                        A positive real number
        computational_budget:               An integer: the maximum number of funciton evaluations
        alpha, gamma, rho, sigma:           positive real numbers that influence how the algorithms behaves
        values:                             previously evaluated function values

    Returns:
        out_1: an approximation of a local optimum of the function
        out_2: number of evaluations of f
    """

    # Pseudo code can be found on: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

    # 0 Order
    if values == []:
        values = np.array([f(vert) for vert in verts])
        calls = calls + len(verts)
    indexes = np.argsort(values)

    x_0 = np.array([0, 0])
    for index in indexes[:-1]:
        x_0 = x_0 + verts[index]
    x_0 = x_0 / (len(verts) - 1)

    x_r = x_0 + alpha * (x_0 - verts[indexes[-1]])
    x_e = x_0 + gamma * (x_r - x_0)
    x_c = x_0 + rho * (verts[indexes[-1]] - x_0)

    # 1 Termination

    if (
        terminate_criterion(verts, f_difference, x_tolerance, y_tolerance)
        or f_difference < y_tolerance
        or calls >= computational_budget
    ):
        return [np.round(verts[indexes[0]]), calls]

    # 3 Reflection
    f_x_r = f(x_r)
    calls += 1
    if values[indexes[0]] <= f_x_r:
        if f_x_r < values[indexes[-2]]:
            f_difference = abs(f_x_r - values[indexes[0]])
            values[indexes[-1]] = f_x_r
            return call_nelder_mead_method(
                f,
                nm_replace_final(verts, indexes, x_r),
                dim,
                f_difference,
                calls,
                terminate_criterion,
                x_tolerance,
                y_tolerance,
                computational_budget,
                alpha,
                gamma,
                rho,
                sigma,
                values,
            )

    # 4 Expansion

    if f_x_r < values[indexes[0]]:
        # x_e = x_0 + gamma * (x_r - x_0)
        f_x_e = f(x_e)
        calls += 1
        if f_x_e < f_x_r:
            f_difference = abs(f_x_e - values[indexes[0]])
            values[indexes[-1]] = f_x_e
            return call_nelder_mead_method(
                f,
                nm_replace_final(verts, indexes, x_e),
                dim,
                f_difference,
                calls,
                terminate_criterion,
                x_tolerance,
                y_tolerance,
                computational_budget,
                alpha,
                gamma,
                rho,
                sigma,
                values,
            )
        else:
            f_difference = abs(f_x_r - values[indexes[0]])
            values[indexes[-1]] = f_x_r
            return call_nelder_mead_method(
                f,
                nm_replace_final(verts, indexes, x_r),
                dim,
                f_difference,
                calls,
                terminate_criterion,
                x_tolerance,
                y_tolerance,
                computational_budget,
                alpha,
                gamma,
                rho,
                sigma,
                values,
            )

    # 5 Contraction

    # x_c = x_0 + rho * (verts[indexes[-1]] - x_0)
    f_x_c = f(x_c)
    if f_x_c < f(verts[indexes[-1]]):
        calls += 1
        f_difference = abs(f_x_c - values[indexes[0]])
        values[indexes[-1]] = f_x_c
        return call_nelder_mead_method(
            f,
            nm_replace_final(verts, indexes, x_c),
            dim,
            f_difference,
            calls,
            terminate_criterion,
            x_tolerance,
            y_tolerance,
            computational_budget,
            alpha,
            gamma,
            rho,
            sigma,
            values,
        )

    # 6 Shrink

    return call_nelder_mead_method(
        f,
        nm_shrink(verts, indexes, sigma),
        dim,
        f_difference,
        calls,
        terminate_criterion,
        x_tolerance,
        y_tolerance,
        computational_budget,
        alpha,
        gamma,
        rho,
        sigma,
    )
