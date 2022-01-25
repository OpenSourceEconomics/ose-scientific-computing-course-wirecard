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







class griewank_info:     ##### This class stores the general information for a griewank function 
    
    def __init__(self,dim,a):   ### arguments are the number of dimensions of the problem and the parameter a
        domain=([-100]*dim,[100]*dim) 
        self.domain=domain   ### returns the domain of the function        
        self.lower_bound=([-100]*dim)  ### returns thw lower bound of the function
        self.upper_bound=([100]*dim)   ### returns the upper bound of the function
        name= 'Griewank Function'
        self.name=name
        problem_solver=np.array([0]*dim)  
        self.solver=problem_solver     ### returns the known solution to the problem
        def function_value(x,a):   
            input = np.array(x)
            sum = (1 / a) * np.dot(input,input)
            prod = np.prod( np.cos( input / np.sqrt( np.arange(1,input.size + 1,1) ) ) )
            out = sum - prod + 2
            return out
        self.solver_function_value=function_value(problem_solver,a)  ### returns the function value of the known solution to the problem
        

        
        
        
class rastrigin_info:     ##### This class stores the general information for a rastrigin function 
    
    def __init__(self,dim,A):   ### arguments are the number of dimensions of the problem and the parameter a
        domain=([-5.12]*dim,[5.12]*dim) 
        self.domain=domain   ### returns the domain of the function        
        self.lower_bound=([-5.12]*dim)  ### returns thw lower bound of the function
        self.upper_bound=([5.12]*dim)   ### returns the upper bound of the function
        name= 'Rastrigin Function'
        self.name=name
        problem_solver=np.array([0]*dim)  
        self.solver=problem_solver     ### returns the known solution to the problem
        def function_value(x,A):
            input = np.array(x)
            n = input.size
            out = A * n + np.sum(input * input - A * np.cos(2 * np.pi * input)) + 1
            return out

        self.solver_function_value=function_value(problem_solver,A)  ### returns the function value of the known solution to the problem
        
    
    
    
class levi_info:     ##### This class stores the general information for a levi_no_13 function 
    
    def __init__(self,dim):   ### arguments are the number of dimensions of the problem and the parameter a
        domain=([-10]*dim,[10]*dim) 
        self.domain=domain   ### returns the domain of the function        
        self.lower_bound=([-10]*dim)  ### returns thw lower bound of the function
        self.upper_bound=([10]*dim)   ### returns the upper bound of the function
        name= 'Levi'
        self.name=name
        problem_solver=np.array([1]*dim)  
        self.solver=problem_solver     ### returns the known solution to the problem
        def function_value(x):
            x = np.asarray_chkfinite(x)
            n = len(x)
            z = 1 + (x - 1) / 4
            return (np.sin( np.pi * z[0] )**2 + np.sum( (z[:-1] - 1)**2 * (1 + 10 * np.sin( np.pi * z[:-1] + 1 )**2 ))
                    + (z[-1] - 1)**2 * (1 + np.sin( 2 * np.pi * z[-1] )**2 ))

            
        self.solver_function_value=function_value(problem_solver)  ### returns the function value of the known solution to the problem
        
        
class rosenbrock_info:     ##### This class stores the general information for a rosenbrock function 
    
    def __init__(self,dim):   ### arguments are the number of dimensions of the problem and the parameter a
        domain=([-100]*dim,[100]*dim) 
        self.domain=domain   ### returns the domain of the function        
        self.lower_bound=([-100]*dim)  ### returns thw lower bound of the function
        self.upper_bound=([100]*dim)   ### returns the upper bound of the function
        name= 'Rosenbrock'
        self.name=name
        problem_solver=np.array([1]*dim)  
        self.solver=problem_solver     ### returns the known solution to the problem
        def function_value(x):
            input = np.array(x)
            if input.size <= 1:
                return 0
            else:
                return (np.sum(100 * (input[1:] - input[:-1] ** 2) ** 2 + (1 - input[:-1]) ** 2) + 1)

        self.solver_function_value=function_value(problem_solver)  ### returns the function value of the known solution to the problem
        

    
class bukin_6_info:     ##### This class stores the general information for a bukin_no_6 function 
    
    def __init__(self,dim):
        dim=2          ### arguments are the number of dimensions of the problem and the parameter a
        domain=([-15,-5],[-3,3]) 
        self.domain=domain   ### returns the domain of the function        
        self.lower_bound=([-15,-3])  ### returns thw lower bound of the function
        self.upper_bound=([-5,3])   ### returns the upper bound of the function
        name= 'Bukin_6'
        self.name=name
        problem_solver=np.array([-10,1])  
        self.solver=problem_solver     ### returns the known solution to the problem
        def function_value(x):
            return [100 * np.sqrt(np.abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * np.abs(x[0] + 10)]
        
        self.solver_function_value=function_value(problem_solver)  ### returns the function value of the known solution to the problem
        
        
class camel_3_info:     ##### This class stores the general information for a three hump camel function 
    
    def __init__(self,dim):
        dim=2              ### arguments are the number of dimensions of the problem and the parameter a
        domain=([-5]*dim,[5]*dim)
        self.domain=domain   ### returns the domain of the function        
        self.lower_bound=([-5]*dim)  ### returns thw lower bound of the function
        self.upper_bound=([5]*dim)   ### returns the upper bound of the function
        name= 'camel_3'
        self.name=name
        problem_solver=np.array([0,0])  
        self.solver=problem_solver     ### returns the known solution to the problem
        def function_value(x):
             return [2 * x[0] ** 2 - 1.05 * x[0] ** 4 + x[0] ** 6 / 6 + np.prod(x) + x[1] ** 2]
        
        self.solver_function_value=function_value(problem_solver)  ### returns the function value of the known solution to the problem
        
class easom_info:     ##### This class stores the general information for a easom function 
    
    def __init__(self,dim):
        dim=2              ### arguments are the number of dimensions of the problem and the parameter a
        domain=([-100]*dim,[100]*dim)
        self.domain=domain   ### returns the domain of the function        
        self.lower_bound=([-100]*dim)  ### returns thw lower bound of the function
        self.upper_bound=([100]*dim)   ### returns the upper bound of the function
        name= 'Easom'
        self.name=name
        problem_solver=np.array([np.pi,np.pi])  
        self.solver=problem_solver     ### returns the known solution to the problem
        def function_value(x):
            return [(-np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2)))]
           
        
        self.solver_function_value=function_value(problem_solver)  ### returns the function value of the known solution to the problem
        
        
class mc_cormick_info:     ##### This class stores the general information for a bukin_no_6 function 
    
    def __init__(self,dim):
        dim=2          ### arguments are the number of dimensions of the problem and the parameter a
        domain=([-1.5,4],[-3,4]) 
        self.domain=domain   ### returns the domain of the function        
        self.lower_bound=([-1.5,-3])  ### returns thw lower bound of the function
        self.upper_bound=([4,4])   ### returns the upper bound of the function
        name= 'Mc_Cormick'
        self.name=name
        problem_solver=np.array([-0.54719,-1.54719])  
        self.solver=problem_solver     ### returns the known solution to the problem
        def function_value(x):
            return [np.sin(np.sum(x)) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1]
            
        self.solver_function_value=function_value(problem_solver)  ### returns the function value of the known solution to the problem
        



class markowitz_info:     ##### This class stores the general information for markowitz function 
    
    def __init__(self,dim,B):
        dim=3          ### arguments are the number of dimensions of the problem and the parameter a
        domain=([0,B],[0,B],[0,B]) 
        self.domain=domain   ### returns the domain of the function        
        self.lower_bound=([0,0,0])  ### returns thw lower bound of the function
        self.upper_bound=([B,B,B])   ### returns the upper bound of the function
        name= 'Markowitz'
        self.name=name
        problem_solver=np.array([0.4411,0.3656,0.1933])  
        self.solver=problem_solver     
        self.solver_function_value=np.array([0.0052820694790865]) 
        
        
        
        
class ackley_info:     ##### This class stores the general information for a griewank function 
    
    def __init__(self,dim):   ### arguments are the number of dimensions of the problem and the parameter a
        domain=([-32.768]*dim,[32.768]*dim) 
        self.domain=domain   ### returns the domain of the function        
        self.lower_bound=([-32.768]*dim)  ### returns thw lower bound of the function
        self.upper_bound=([32.768]*dim)   ### returns the upper bound of the function
        name= 'Ackley Function'
        self.name=name
        problem_solver=np.array([0]*dim)  
        self.solver=problem_solver     ### returns the known solution to the problem
        def function_value(x):   
            a=20
            b=0.2
            c=2*np.pi
            x = np.asarray_chkfinite(x) 
            n = len(x)
            s1 = np.sum( x**2 )
            s2 = np.sum( cos( c * x ))
            return -a*np.exp( -b*np.sqrt( s1 / n )) - np.exp( s2 / n ) + a + np.exp(1)

        self.solver_function_value=function_value(problem_solver)  ### returns the function value of the known solution to the problem
        
        
        
        
class schwefel_info:     ##### This class stores the general information for a griewank function 
    
    def __init__(self,dim):   ### arguments are the number of dimensions of the problem and the parameter a
        domain=([-500]*dim,[500]*dim) 
        self.domain=domain   ### returns the domain of the function        
        self.lower_bound=([-500]*dim)  ### returns thw lower bound of the function
        self.upper_bound=([500]*dim)   ### returns the upper bound of the function
        name= 'Schwefel Function'
        self.name=name
        problem_solver=np.array([420.968746]*dim)  
        self.solver=problem_solver     ### returns the known solution to the problem
        def function_value(x):
            x = np.asarray_chkfinite(x)
            n = len(x)
            return 418.9829*n - np.sum( x * np.sin( np.sqrt( np.abs( x ))))
        
        self.solver_function_value=function_value(problem_solver)  ### returns the function value of the known solution to the problem
        
        
class zakharov_info:     ##### This class stores the general information for a griewank function 
    
    def __init__(self,dim):   ### arguments are the number of dimensions of the problem and the parameter a
        domain=([-50]*dim,[50]*dim) 
        self.domain=domain   ### returns the domain of the function        
        self.lower_bound=([-50]*dim)  ### returns thw lower bound of the function
        self.upper_bound=([50]*dim)   ### returns the upper bound of the function
        name= 'Zakharov Function'
        self.name=name
        problem_solver=np.array([0]*dim)  
        self.solver=problem_solver     ### returns the known solution to the problem
        def function_value(x):
            x = np.asarray_chkfinite(x)
            n = len(x)
            j = np.arange( 1., n+1 )
            s2 = np.sum( j * x ) / 2
            return np.sum( x**2 ) + s2**2 + s2**4


        self.solver_function_value=function_value(problem_solver)  ### returns the function value of the known solution to the problem
        
        
        
        

        
        
        
        
        

        




