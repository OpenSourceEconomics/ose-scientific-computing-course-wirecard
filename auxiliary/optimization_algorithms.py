import autograd.numpy as np
from autograd import grad, jacobian


# In this File we are going to implement the optimization algorithms we study in our assignment

# To get some practice with implementing optimization algorithms I implemented the 1 Dimensional Newton Method,
# and a naive optimization algorithms that uses the multidimensional newton method


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
    # print("find_starting point bekommt n == ", n)
    A = np.random.rand(k, n)
    B = [f(domain[0] + (domain[1] - domain[0]) * x) for x in A]
    index = np.where(B == np.amin(B))
    return A[index][0]


def initial_simplex(dim, domain):
    """Return a dim- dimensional simplex within the cube domain^n
    Args:
        dim:           the dimension we are working with
        domain:        edges of the domain

    Returns:
        out:           the verticies of the simplex in an dim+1 dimensional array

    """
    A = np.random.rand(dim + 1, dim)
    A = [domain[0] + x * (domain[1] - domain[0]) for x in A]

    return A


def newton_method(f, df, x_n, eps=10 ** (-6), n=1000):
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
    # print("newton method bekommt als x_n: ", x_n)
    f_xn = f(x_n)
    while np.linalg.norm(f_xn) > eps and n > 0:
        sol = np.linalg.solve(df(x_n), -f_xn)
        x_n = x_n + sol
        # print("x_n equals to: ", x_n)
        # np.linalg.lstsq can deal with non invertibel matrices
        # x_n = x_n + np.linalg.solve(df(x_n), -f_xn)
        f_xn = f(x_n)
        n = n - 1
    # print("n equals: ",n)
    if np.linalg.norm(f_xn) < eps:
        return x_n
    else:
        return "Didnt converge."


def centroid(verts):
    c = np.zeros(len(verts[0]))
    for vert in verts:
        c = c + vert
    c = (1 / len(verts)) * c
    return c


def new_shrink(verts, indexes):
    x_l = verts[indexes[0]]
    out = np.array([(vert + x_l) * 0.5 for vert in verts])
    return out


def accept(verts, x_new):
    out = np.array(verts[:-1] + [x_new])
    return out


def new_nelder_mead(f, verts):
    values = np.array([f(vert) for vert in verts])
    indexes = np.argsort(values)
    x_l = verts[indexes[0]]
    x_h = verts[indexes[-1]]
    x_n = verts[indexes[-2]]
    f_l = f(x_l)
    f_h = f(x_h)
    f_n = f(x_n)
    c = centroid(verts)
    x_r = c + (c - x_h)
    f_r = f(x_r)
    if nm_terminate:
        return c
    elif f_r < f_l:
        x_e = c + 2 * (c - x_h)
        f_e = f(x_e)
        if f_e < f_l:
            verts = accept(verts, x_e)
            # accept x_e
        else:
            verts = accept(verts, x_r)
            # accept x_r
    elif f_r < f_n:
        verts = accept(verts, x_r)
        # accept x_r
    elif f_r < f_h:
        x_c = c + 0.5 * (c - x_h)
        f_c = f(x_c)
        if f_c <= f_r:
            verts = accept(verts, x_c)
            # accept x_c
        else:
            verts = new_shrink(verts, indexes)
    else:
        x_c = c + 0.5 * (x_h - c)
        f_c = f(x_c)
        if f_c < f_h:
            verts = accept(verts, x_c)
            # accept x_c
        else:
            verts = new_shrink(verts, indexes)
    return new_nelder_mead(f, verts)


def nelder_mead_method(f, verts, dim, alpha=1, gamma=2, rho=0.5, sigma=0.5):
    # Pseudo code can be found on: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

    # 0 Order
    print("nm called with verts = ", verts)

    values = np.array([f(vert) for vert in verts])
    indexes = np.argsort(values)
    x_0 = np.array([0, 0])
    for index in indexes[:-1]:
        x_0 = x_0 + verts[index]
    x_0 = x_0 / (len(verts) - 1)

    x_r = x_0 + alpha * (x_0 - verts[indexes[-1]])
    print("x_r equals: ", x_r)
    x_e = x_0 + gamma * (x_r - x_0)
    x_c = x_0 + rho * (verts[indexes[-1]] - x_0)
    for i in indexes:
        print("verts: ", verts[i], "values: ", values[i])
    print("x_0: ", x_0, "value: ", f(x_0))
    print("x_r: ", x_r, "value: ", f(x_r))
    print("x_e: ", x_e, "value: ", f(x_e))
    print("x_c: ", x_c, "value: ", f(x_c))

    # 1 Termination

    if nm_terminate(verts):
        print("Termination")
        return verts[indexes[0]]

    # 2 Calculate x_0

    # x_0 = np.sum(np.array([verts[i] for i in indexes[:-1]])) / (len(verts) - 1)

    # 3 Reflection

    # x_r = x_0 + alpha * (x_0 - verts[indexes[-1]])
    if values[indexes[0]] <= f(x_r):
        print("entered if")
        if f(x_r) < values[indexes[-2]]:
            print("Reflection")
            return nelder_mead_method(
                f, nm_replace_final(verts, indexes, x_r), dim, alpha, gamma, rho, sigma
            )

    # 4 Expansion

    if f(x_r) < values[indexes[0]]:
        # x_e = x_0 + gamma * (x_r - x_0)
        if f(x_e) < f(x_r):
            print("Expansion 1")
            return nelder_mead_method(
                f, nm_replace_final(verts, indexes, x_e), dim, alpha, gamma, rho, sigma
            )
        else:
            print("Expansion 2")
            return nelder_mead_method(
                f, nm_replace_final(verts, indexes, x_r), dim, alpha, gamma, rho, sigma
            )

    # 5 Contraction

    # x_c = x_0 + rho * (verts[indexes[-1]] - x_0)
    if f(x_c) < f(verts[indexes[-1]]):
        print("Contraction")
        return nelder_mead_method(
            f, nm_replace_final(verts, indexes, x_c), dim, alpha, gamma, rho, sigma
        )

    # 6 Shrink

    return nelder_mead_method(
        f, nm_shrink(verts, indexes, sigma), dim, alpha, gamma, rho, sigma
    )


def nm_terminate(verts):
    eps = 0
    for vert in verts:
        eps += abs(np.linalg.norm(vert - verts[0]))
    print("Summed distance = ", eps)
    if eps < 1e-4:
        return True
    if eps > 50:
        return True
    else:
        return False


def nm_replace_final(verts, indexes, x_new):  # passed pytest
    new_verts = []
    for i in range(len(verts)):
        new_verts.append(verts[i])
    new_verts[indexes[-1]] = x_new
    new_verts = np.array(new_verts)
    return new_verts


def nm_shrink(verts, indexes, sigma):  # passed pytest
    new_verts = []
    for i in range(indexes.size):
        new_verts.append(verts[indexes[0]] + sigma * (verts[i] - verts[indexes[0]]))
    new_verts = np.array(new_verts)
    return new_verts


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


if __name__ == "__main__":

    test_finding_starting_point = False
    test_initial_simplex = False

    if test_finding_starting_point:
        print(find_starting_point(lambda a: a[0] + a[1], [4, 6], 2))
    if test_initial_simplex:
        simplex = initial_simplex(3, [5, 6])
        for vert in simplex:
            print(vert)
