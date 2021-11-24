import numpy as np  
import matplotlib.pyplot as plt
import random

from mpl_toolkits.mplot3d import Axes3D


# functions to be tested
from test_functions import rastrigin
from test_functions import rosenbrock
from test_functions import levi_no_13
from test_functions import griewank



if __name__ == "__main__":

    x_interval = (-5.12,5.12)
    y_interval = (-5.12,5.12)
    x_points = np.linspace(x_interval[0], x_interval[1],100)
    y_points = np.linspace(y_interval[0], y_interval[1],100)

    X, Y = np.meshgrid(x_points, y_points)

    rastrigin_vectorized = np.vectorize(lambda a,b : rastrigin([a,b]))
    
    Z = rastrigin_vectorized(X,Y)

    plt.figure(figsize=(20,10))
    ax = plt.axes(projection = '3d')

    ax.plot_surface(X, Y, Z)
    ax.set(xlabel='x', ylabel='y', zlabel='f(x, y)', title='f = rastragin function')
    plt.show()