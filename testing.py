import numpy as np  
import matplotlib.pyplot as plt
import random

from mpl_toolkits.mplot3d import Axes3D


# functions to be tested
from test_functions import rastrigin
from test_functions import rosenbrock
from test_functions import levi_no_13
from test_functions import griewank

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

    plot_test_function((-5.12,5.12), rastrigin, 'rastrigin')
    plot_test_function((-100,100), griewank, 'griewank')
    plot_test_function((-10,10), levi_no_13, 'Levi no. 13')
    plot_test_function((-100,100), rosenbrock, 'rosenbrock')