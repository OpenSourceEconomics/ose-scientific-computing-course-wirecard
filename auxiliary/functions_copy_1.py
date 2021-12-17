from numpy import *
import nlopt
import numpy as np
import matplotlib.pyplot as plt
import numbers
import math
import pandas as pd
import random
import autograd.numpy as ag
from autograd import grad
from mpl_toolkits.mplot3d import Axes3D
from numpy.lib.function_base import vectorize
from autograd import value_and_grad

##### Define the test problems discussed in Arnoud, Guvenen and Kleineberg as classes
##### These allow us to evaluate the function for a vector x, and a value a -> Guvenen et al. specified a=200 in their Analysis -> we do this too


class griewank:
    def __init__(self, x, a):
        self.arg = x
        self.dim = x.size  # allows to check the dimension of the function

    def function_value(
        x, a
    ):  # function that gives us the function value of the griewank function as specified in Guvenen et al.
        input = np.array(x)
        sum = (1 / a) * np.dot(input, input)
        prod = np.prod(np.cos(input / np.sqrt(np.arange(1, input.size + 1, 1))))
        out = sum - prod + 2
        return out

    self.function_val = function_value(x, a)

    # returns the function value for the griewank function evaluated at x
    # domain=([-100]*self.dim,[100]*self.dim)
    # self.domain=domain
    # self.lower_bound=([-100]*self.dim)
    # self.upper_bound=([100]*self.dim)
    # name= 'Griewank Function'
    # self.name=name
    # problem_solver=np.array([0]*self.dim) # best known global minimum
    # self.solver=problem_solver
    # self.solver_function_value=function_value(problem_solver,a)


class griewank_info:  ##### This class stores the general information for a griewank function
    def __init__(
        self, dim, a
    ):  ### arguments are the number of dimensions of the problem and the parameter a
        domain = ([-100] * dim, [100] * dim)
        self.domain = domain  ### returns the domain of the function
        self.lower_bound = [-100] * dim  ### returns thw lower bound of the function
        self.upper_bound = [100] * dim  ### returns the upper bound of the function
        name = "Griewank Function"
        self.name = name
        problem_solver = np.array([0] * dim)
        self.solver = problem_solver  ### returns the known solution to the problem

        def function_value(x, a):
            input = np.array(x)
            sum = (1 / a) * np.dot(input, input)
            prod = np.prod(np.cos(input / np.sqrt(np.arange(1, input.size + 1, 1))))
            out = sum - prod + 2
            return out

        self.solver_function_value = function_value(
            problem_solver, a
        )  ### returns the function value of the known solution to the problem


##### Now we define the griewank function such that it fits the interface from nlopt


def problem_griewank(x, grad):
    return (
        (1 / 200) * np.dot(x, x)
        - np.prod(np.cos(x / np.sqrt(np.arange(1, x.size + 1, 1))))
        + 2
    )  #### returns the griewank problem


### x is the input vector
### grad is the gradient vector -> for derivative free algorithms we do not need this argument
### nlopt algorithms also work if grad=NONE
### Most of our algorithms are derivative free except for StoGO


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
