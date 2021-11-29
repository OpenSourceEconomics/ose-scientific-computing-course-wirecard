import autograd.numpy as np  
import matplotlib.pyplot as plt
import math
import random

from mpl_toolkits.mplot3d import Axes3D
from numpy.lib.function_base import vectorize


# functions to be tested
from test_functions import rastrigin
from test_functions import rosenbrock
from test_functions import levi_no_13
from test_functions import griewank
from optimization_algorithms import find_starting_point, naive_optimization, newton_method_1D
from optimization_algorithms import first_derivative_1D
from optimization_algorithms import newton_method


def plot_test_function(domain, function, name_of_function = 'some function'): 
    """Plot a 3d graph of a function. 

    Args:
        domain:                     domain \subset \R of the function 
        function:                   a function that returns a value for a input array [a,b] where a,b are in the domain
        name_of_function (string) : the name of the function
        

    Returns:
        a 3d plot of the function

    """
    x_points = np.linspace(domain[0], domain[1],100)
    y_points = np.linspace(domain[0], domain[1],100)

    X, Y = np.meshgrid(x_points, y_points)

    rastrigin_vectorized = np.vectorize(lambda a,b : function([a,b]))
    
    Z = rastrigin_vectorized(X,Y)

    plt.figure(figsize=(20,10))
    ax = plt.axes(projection = '3d')

    ax.plot_surface(X, Y, Z)
    ax.set(xlabel='x', ylabel='y', zlabel='f(x, y)', title='f = ' + name_of_function)
    plt.show()
    pass


if __name__ == "__main__":

    plot_test_functions = False
    test_newton_1D = False
    test_newton = True


    if plot_test_functions == True:
        plot_test_function((-5.12,5.12), rastrigin, 'rastrigin')
        plot_test_function((-100,100), griewank, 'griewank')
        plot_test_function((-10,10), levi_no_13, 'Levi no. 13')
        plot_test_function((-100,100), rosenbrock, 'rosenbrock')
    
    if test_newton_1D == True:
        print("One root of x^2 + 1 is at x == ", newton_method_1D(lambda a : a**2 - 4, 6))
        f = lambda a : (a-4)**2 + 1
        df_2 = lambda b : 2 * b - 8
        df = lambda c : first_derivative_1D(c, f, eps = 10**-8)
        print("Min of (x-4)^2 + 1 is at x == ", newton_method_1D(df_2,1))
        print("Min of (x-4)**2 + 1 is at x == ", newton_method_1D(df, 1))

    if test_newton == True:
        #function whose minimum we want to find:
        f = lambda a : (a[0] + a[1])**2 - 0.5 * a[1]**2 + 1
        #thus we want to find the zeros of its derivative:
        df = lambda b : np.array([2 * b[0] + 2 * b[1] + 1, 2 * b[0] + b[1]])
        #its derivatives Jacobian is given by: 
        J = lambda c : np.array([[2,2],
                                 [2,1]])
        print("x -> [2 * x_1 + 2 * x_2 + 1, 2 * x_1 + x_2] hat eine Nulstelle bei: ", newton_method(df,J, np.array([20.234,100.391])))
    
        f_2 = lambda a : a # not in the mood to solve this differential equation
        df_2 = lambda a : np.array([a[0]**2 - a[1] + a[0] * np.cos(np.pi * a[0]),a[0]*a[1] + math.exp(-a[1]) - 1 / a[0]])
        J_2 = lambda a : np.array([[2*a[0] + np.cos(np.pi*a[0]) - np.pi*a[0]*np.sin(np.pi*a[0]), -1],
                                  [a[1] + 1 / (a[0]**2), a[0] - math.exp(-a[1])]])
                    
        
        

        
                                
        print("df_2 hat eine Nulstelle bei: ", newton_method(df_2,J_2, np.array([2,1])))

        print(" x -> (x1 + x2)^2 - 0.5 x_2^2 + 1 hat ein optimum bei: ", naive_optimization(f, 2, np.array([100.23, 6.55])))

        f_3 = lambda a : rosenbrock(a)   # n - dimensionale rosenbrock
        print("Die 30-dim Rosenbrock funktion hat ein Optimum bei: ", naive_optimization(f_3,12,np.array([89.,21.,43.,55.,12.,13.,89.,21.,43.,55.,12.,13.,89.,21.,43.,55.,12.,13.,89.,21.,43.,55.,12.,13.,89.,21.,43.,55.,12.,13.])))

        
        f_4 = lambda a : griewank(a)    # 2 dimensionale griewank
        print("Die 2-dim griewank funktion hat ein Optimum bei: ", naive_optimization(f_4,2,[-5,5]))

        f_5 = lambda a : rastrigin(a)   # 2 dimensional rastrigin
        print("Die 2-dim rastrigin function hat ein Optimum bei: ", naive_optimization(f_5,2,[-5,5]))

        



