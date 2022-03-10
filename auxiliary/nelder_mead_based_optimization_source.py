import numpy as np


def initial_simplex(dim, domain, eps=0.5):
    """Return a dim- dimensional simplex within the cube domain^n
    Args:
        dim:           the dimension we are working with
        domain:        edges of the domain
        eps:           parameter to control the size of the simplex

    Returns:
        A:           the verticies of the simplex in an dim+1 dimensional array

    """
    # get random points
    A = np.random.rand(dim + 1, dim) * eps
    # compute the simplex
    A = [domain[0] + x * (domain[1] - domain[0]) for x in A]

    return A


# this function is not used yet but could be integrated later to give the optimization algorithms more flexibility
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


# this function is not used yet but could be integrated later to give the optimization algorithms more flexibility
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


# checks if the termination criterion in x is fullfilled
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
    # compute deviation
    for vert in verts:
        eps += abs(np.linalg.norm(vert - verts[0]))
    eps = eps / len(verts)

    # decide wether to stop or continue search for optimum
    if eps < x_tolerance:
        return True
    if eps > 50:
        return True
    else:
        return False


# checks if the termination criterion in y is fullfilled
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
    # decide wether to stop or continue search for optimum
    if f_difference < y_tolerance:
        return True
    else:
        return False


# function to replace the 'worst' verticy
def nm_replace_final(verts, indexes, x_new):
    """Replaces the verticy with the highest associated function value with x_new.
    Args:
        verts:      Array of verticies
        indexes:    Arrays if intergers ordering verts according to assiciated function values
        x_new:      New verticy  to be added to verts

    Returns:
        out:        new simplex

    """
    # intitilize variables
    new_verts = []
    # copy all old verts into the new array
    for i in range(len(verts)):
        new_verts.append(verts[i])
    # replace the worst vert
    new_verts[indexes[-1]] = x_new
    new_verts = np.array(new_verts)
    return new_verts


# the shrink action of the nelder-mead method
def nm_shrink(verts, indexes, sigma):
    """Shrinks the simplex according to the definition of the nelder-mead method.
    Args:
        verts:      Array of verticies
        indexes:    Arrays if intergers ordering verts according to assiciated function values
        sigma:      a parameter important to compute the shrinked simplex

    Returns:
        out:       new simplex

    """
    # set best vert
    v_0 = verts[indexes[0]]
    # compute other verts
    for i in indexes[1:]:
        verts[i] = v_0 + sigma * (verts[i] - v_0)
    return verts


# the nelder_mead_method implemented recursivly
def call_nelder_mead_method(
    f,
    verts,
    x_tolerance=1e-6,
    y_tolerance=1e-6,
    computational_budget=1000,
    f_difference=10,
    calls=0,
    terminate_criterion=terminate_criterion_x,
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
        return [np.array(np.round(verts[indexes[0]])), calls]

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
                x_tolerance,
                y_tolerance,
                computational_budget,
                f_difference,
                calls,
                terminate_criterion,
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
                x_tolerance,
                y_tolerance,
                computational_budget,
                f_difference,
                calls,
                terminate_criterion,
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
                x_tolerance,
                y_tolerance,
                computational_budget,
                f_difference,
                calls,
                terminate_criterion,
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
            x_tolerance,
            y_tolerance,
            computational_budget,
            f_difference,
            calls,
            terminate_criterion,
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
        x_tolerance,
        y_tolerance,
        computational_budget,
        f_difference,
        calls,
        terminate_criterion,
        alpha,
        gamma,
        rho,
        sigma,
        values,
    )
