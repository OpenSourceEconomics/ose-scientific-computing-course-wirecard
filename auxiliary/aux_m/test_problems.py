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


####### Functions that work for all algorithms 



def problem_griewank(x): 
    return (1 / 200) * np.dot(x,x)- np.prod( np.cos( x / np.sqrt( np.arange(1,x.size + 1,1) ) ) )+2    #### Function is specified as in Guvenen et al.



def problem_rastrigin(x):
    input = np.array(x)
    n = input.size
    out = 10 * n + np.sum(input * input - 10 * np.cos(2 * np.pi * input)) + 1  ### code A=10
    return out

    


def problem_levi(x):
    x = np.asarray_chkfinite(x)
    n = len(x)
    z = 1 + (x - 1) / 4
    return (np.sin( np.pi * z[0] )**2 + np.sum( (z[:-1] - 1)**2 * (1 + 10 * np.sin( np.pi * z[:-1] + 1 )**2 ))
            + (z[-1] - 1)**2 * (1 + np.sin( 2 * np.pi * z[-1] )**2 ))

    
    
def problem_rosenbrock(x):
    input = np.array(x)
    if input.size <= 1:
        return 0
    else:
        return (np.sum(100 * (input[1:] - input[:-1] ** 2) ** 2 + (1 - input[:-1]) ** 2) + 1)
    
    
    
def problem_bukin_6(x,grad):
    
    return 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * np.abs(x[0] + 10)


def problem_camel_3(x,grad):
    
    return 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + x[0] ** 6 / 6 + np.prod(x) + x[1] ** 2

def problem_easom(x,grad):
    
    return (-np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2)))

def problem_mc_cormick(x,grad):
    
    return np.sin(np.sum(x)) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1




def problem_application_correct_1(x,grad):
    
    #### define problem of the application 3 dimensional case
    B=1
    lambda_param=1
    alpha_1=0.0427
    alpha_2=0.0015
    alpha_3=0.0285
    sigma_x_1=0.01 ### square std deviation to get variance
    sigma_x_2=0.01089936
    sigma_x_3=0.01990921
    sigma_x_12=0.0018
    sigma_x_13=0.0011
    sigma_x_23=0.0026
    
    if grad.size > 0:
        grad[0]=sigma_x_1*x[0]*2+2*sigma_x_12*x[1]+2*sigma_x_13*x[2]
        grad[1]=sigma_x_2*x[1]*2+2*sigma_x_12*x[0]+2*sigma_x_23*x[2]
        grad[2]=sigma_x_3*x[2]*2+2*sigma_x_13*x[0]+2*sigma_x_23*x[1]
    
    exp_returns=-alpha_1*x[0]-alpha_2*x[1]-alpha_3*x[2]
    
    disutility_volatility=(sigma_x_1*(x[0]**2))+(x[1]**2)*sigma_x_2+(x[2]**2)*sigma_x_3+2*sigma_x_12*x[0]*x[1]+2*sigma_x_13*x[0]*x[2]+2*x[1]*x[2]*sigma_x_23
    
    #### disutility is coded without lambda
    return disutility_volatility
    

    
    





