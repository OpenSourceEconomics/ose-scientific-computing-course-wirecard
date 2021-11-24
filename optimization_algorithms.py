import numpy as np

# In this File we are going to implement the optimization algorithms we study in our assignment

# To get some practice with implementing optimization algorithms I am going to work on an simple
# version of the Newton procedure for function of the form f: \R \to \R

# After that I want to implement a multidimensional version but that will still take some time i guess

def first_derivative_(x, f, eps = 10**(-6)): 
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

def newton_method_1D(f, x_0, eps_newton = 10** (-6), eps_derivative = 10** (-6), n = 100000):
    previous, next = x_0
    df = lambda a : first_derivative(a,f,eps_derivative)
    for i in range(1,n):
        next = previous + f(previous) / df(previous)
        previous = next 
        if df(next) < eps_newton:
            break
    if i == n-1: 
        return None
    else: 
        return previous

