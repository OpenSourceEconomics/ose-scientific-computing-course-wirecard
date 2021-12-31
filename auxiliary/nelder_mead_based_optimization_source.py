import numpy as np


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


# TODO - write a nelder-mead routine for michael that takes f, startpoint x_0
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

# Maybe we could implement the nelder-mead-method as a class such that we can safe, call and change the values alpha, gamma, rho and sigma


def nelder_mead_method(f, verts, dim, alpha=1, gamma=2, rho=0.5, sigma=0.5):
    # Pseudo code can be found on: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

    # 0 Order
    values = np.array([f(vert) for vert in verts])
    indexes = np.argsort(values)

    x_0 = np.array([0, 0])
    for index in indexes[:-1]:
        x_0 = x_0 + verts[index]
    x_0 = x_0 / (len(verts) - 1)

    x_r = x_0 + alpha * (x_0 - verts[indexes[-1]])
    x_e = x_0 + gamma * (x_r - x_0)
    x_c = x_0 + rho * (verts[indexes[-1]] - x_0)

    # 1 Termination

    if nm_terminate(verts):
        return np.round(verts[indexes[0]])

    # 3 Reflection
    if values[indexes[0]] <= f(x_r):
        if f(x_r) < values[indexes[-2]]:
            return nelder_mead_method(
                f, nm_replace_final(verts, indexes, x_r), dim, alpha, gamma, rho, sigma
            )

    # 4 Expansion

    if f(x_r) < values[indexes[0]]:
        # x_e = x_0 + gamma * (x_r - x_0)
        if f(x_e) < f(x_r):
            return nelder_mead_method(
                f, nm_replace_final(verts, indexes, x_e), dim, alpha, gamma, rho, sigma
            )
        else:
            return nelder_mead_method(
                f, nm_replace_final(verts, indexes, x_r), dim, alpha, gamma, rho, sigma
            )

    # 5 Contraction

    # x_c = x_0 + rho * (verts[indexes[-1]] - x_0)
    if f(x_c) < f(verts[indexes[-1]]):
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
    # print("Summed distance = ", eps)
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