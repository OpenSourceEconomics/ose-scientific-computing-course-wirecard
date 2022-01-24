# Until we have a clearer image of how our project is going to unfold,
# I will be commiting my programming-progress in this File
from numpy import *
import nlopt
import numpy as np
import matplotlib.pyplot as plt
import numbers
import math
import pandas as pd
import random
import autograd.numpy as ag
from autograd import grad, value_and_grad
from mpl_toolkits.mplot3d import Axes3D
from numpy.lib.function_base import vectorize


from functions import (
    rastrigin_instance,
    griewank_instance,
    levi_no_13_instance,
    rosenbrock_instance,
)

# a very naive routine to approcximate the derivatives of 1 dimensional functions
def first_derivative_1D(x, f, eps=10 ** (-6)):
    """Return an approximation for the derivative of f at x.

    Args:
        x:   a number within the domain of f
        A:   a function that maps a subset of \R to \R
        eps: a scale to controll the accuracy of the approximation


    Returns:
        out: an approximation of the value of the derivative of f at x

    """
    out = (f(x + eps) - f(x - eps)) / eps
    return out


# 1 dimensional newton method that i impememnted to get familiar to the problem
def newton_method_1D(f, x_n, eps_newton=10 ** (-6), eps_derivative=10 ** (-6), n=1000):
    """Return a candidate for a root of f, if the newton method starting at x_n converges.

    Args:
        f:              a function from \R to \R whose root we want to find
        x_n:            a number within the domain of f from which to start the iteration
        eps_newton:     sensitivity of of the root finding process
        eps_derivative: sensitivity of the derivative approximation
        n:              maximum of iterations before stopping the procedure



    Returns:
        out: either an approximation for a root or a message if the procedure didnt converge

    """
    df = lambda a: first_derivative_1D(a, f, eps_derivative)
    for _i in range(1, n):
        x_n = x_n - (f(x_n) / df(x_n))
        if np.abs(f(x_n)) < eps_newton:
            break
    # print("Ran through: ",i, " times.")

    if i > n - 2:
        return "Didnt converge"
    else:
        return x_n


# this is a function that michael computed
def get_starting_points(n, problem_info_object, p):

    ### n: number of desired dimensions of the problem
    ### problem_info_object: object that contains the known information of the problem e.g.: g_1=griewank_info(n,a=200)
    ### p: desired number of starting points you want to draw

    ## as Guvenen et al. do not specify how they generate the random starting points I will choose a method
    ### Method:
    # as the starting point has to be a vector fo dimension = dimension of the function, I draw every coordinate
    # for the vector from a uniform distribution
    # repeat this until you get 100 vectors of dimension = dim of function which are randomly generated
    data = []

    lower_b = problem_info_object.lower_bound
    upper_b = problem_info_object.upper_bound

    for i in range(n):
        v = np.random.uniform(lower_b[i], upper_b[i], p)
        data.append(v)
    df = pd.DataFrame(data)
    return df.transpose()


# The following is a function that michael implemented
#### first define a function that performs the optimization routine for 100 different starting points
#### starting points are randomly generated
#### do this also for tik tak

#### not yet finished


def minimization_guvenen(
    x_tol_abs, f_tol_abs, problem, computational_budget, algo, x_0, n, problem_info
):

    ##################################     Input that need to be specified ################################
    #######################################################################################################

    ## x_tol: is the absolute Tolerance allowed
    ## f_tol: tolerance in function value allowed
    ## problem: we need to specify the objective function
    ## computational budget: is a vector that contains different computational budgets between 0 and 10^5
    ## algo: specify the algorithm you want to use from the nlopt library -> argument has to have the form:
    ######## nlopt.algorithm_name e.g. nlopt.GN_ISRES for ISRES Algorithm
    ## algorithm: specify the algorithm
    ## x_0: contains the randomly generated starting points -> pandas dataframe containing starting values
    ## n: number of dimensions the problem should have
    ## problem_info: object that that contains known information about the objective function
    ## as for example the domain
    ## the best solver
    ## function value of the best solver etc
    ### If you want to stop the optimization routine when the x_tol_abs for convergence is met
    ########   -> plug in -inf for f_tol_abs
    ##### If you want to stop the optimization routine when the f_tol_abs convergence criterion is met specify:
    ######## -> x_tol_abs=-inf

    ######################################      Output       ################################################

    #### returns a dataframe containing:
    #### a vector of the optimizer -> first n columns # coordinate vector
    #### the function value of the optimizer -> next column                ##### this is done 100 times
    #### number of function evaluations -> next column                     #### for all 100 starting points
    ### accuracy measures as specified in Guvenen et al.

    # np.set_printoptions(precision=20)

    global_optimum = nlopt.opt(algo, n)
    global_optimum.set_lower_bounds(problem_info.lower_bound)
    global_optimum.set_upper_bounds(problem_info.upper_bound)
    global_optimum.set_min_objective(problem)
    global_optimum.set_xtol_abs(x_tol_abs)
    global_optimum.set_ftol_abs(f_tol_abs)
    global_optimum.set_maxeval(computational_budget)

    df = []

    for i in range(len(x_0)):

        # start_point_i=np.array(x_0.iloc[i])
        optimizer = global_optimum.optimize(np.array(x_0.iloc[i]))
        function_val = problem_griewank(optimizer, grad)
        num_evals = np.array(global_optimum.get_numevals())
        #### define accuracy measures

        information = np.hstack((optimizer, function_val, num_evals))
        df.append(information)

    dframe = pd.DataFrame(df)

    return dframe


# This was an implementation of the nelder mead method i considered using for a while
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


# this was an implementeation of the newton methid i considered using for a while
def newton_method_old(f, df, x_n, eps=10 ** (-16), n=1000):
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
    count_calls = 1
    while np.linalg.norm(f_xn) > eps and n > 0:
        sol = np.linalg.solve(df(x_n), -f_xn)
        count_calls = count_calls + 1
        x_n = x_n + sol
        f_xn = f(x_n)
        count_calls = count_calls + 1
        n = n - 2
    return x_n, count_calls


# this was an old implementation of the nelder-mead method, we are not using anymore


def nelder_mead_method(f, verts, dim, alpha=1, gamma=2, rho=0.5, sigma=0.5):
    """Return an approximation of a local optimum.

    Args:
        f:                                  a real valued function
        starting point:                     a point within the domain of f around which the approximation starts
        stopping_tolerance_xvalue:          the tolerance of the stopping criterion in the x argument
        stopping_tolerance_functionvalue:   the tolerance of the stopping criterion in the function value
        computational_budget:               maximal number of function calls after which the algortithm terminates

    Returns:
        out_1: an approximation of a local optimum of the function
        out_2: number of evaluations of f
    """

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


if __name__ == "__main__":
    pass
