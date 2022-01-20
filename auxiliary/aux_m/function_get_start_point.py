from numpy import *
import pandas as pd
import random
import nlopt
import numpy as np
import matplotlib.pyplot as plt
import numbers
import math
import random
import autograd.numpy as ag
from autograd import grad
from mpl_toolkits.mplot3d import Axes3D
from numpy.lib.function_base import vectorize
from autograd import value_and_grad
np.set_printoptions(precision=20)
pd.set_option("display.precision", 14)





def get_starting_points(n,problem_info_object,p):
    
    
    ### n: number of desired dimensions of the problem
    ### problem_info_object: object that contains the known information of the problem e.g.: g_1=griewank_info(n,a=200)
    ### p: desired number of starting points you want to draw
    
     ## as Guvenen et al. do not specify how they generate the random starting points I will choose a method
       ### Method:
        # as the starting point has to be a vector fo dimension = dimension of the function, I draw every coordinate
        # for the vector from a uniform distribution
        # repeat this until you get 100 vectors of dimension = dim of function which are randomly generated
    data=[]
    
    lower_b=problem_info_object.lower_bound
    upper_b=problem_info_object.upper_bound
    
    for i in range(n):
        v=np.random.uniform(lower_b[i],upper_b[i],p)
        data.append(v)
    df=pd.DataFrame(data)
    return df.transpose()



def get_start_points_application(alpha_dirichlet,p,B):
    
     ## alpha dirichlet is a vector that contains alphas for dirichlet distribution
    ## this also stores the dimension of the problem
    ## p is the number of start points we want to generate
    ## B is the budget we consider
    ## function works
    data=[]
    
    for i in range(p):
        
        vector_dirichlet=np.random.dirichlet(alpha_dirichlet,1)
        vector_budget_adjusted=vector_dirichlet*B
        data.append(vector_budget_adjusted)
    
    df=pd.DataFrame(np.concatenate(data))
    return df
    
    
    
    






