import autograd.numpy as np
from autograd import grad, jacobian


# In this File we are going to implement the optimization algorithms we study in our assignment

# To get some practice with implementing optimization algorithms I am going to work on an simple
# version of the Newton procedure for function of the form f: \R \to \R

# After that I want to implement a multidimensional version but that will still take some time i guess

def find_starting_point(f,domain,k = 100):
    print("k equals to: ",k)
    X = np.linspace(domain[0],domain[1],k)
    Y = np.linspace(domain[0],domain[1],k)
    x_0 = [1.,0.]
    f_x0 = f(x_0)
    for x in X:
        for y in Y: 
            if f([x,y]) < f_x0:
                x_0 = [x,y]
                
                f_x0 = f([x,y])
    
    
    return x_0

def first_derivative_1D(x, f, eps = 10**(-6)): 
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

def newton_method_1D(f, x_n, eps_newton = 10** (-6), eps_derivative = 10** (-6), n = 1000):
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
    df = lambda a : first_derivative_1D(a,f,eps_derivative)
    for i in range(1,n):
        x_n = x_n - (f(x_n) / df(x_n))
        if np.abs(f(x_n)) < eps_newton:
            break
    #print("Ran through: ",i, " times.")
    
    if i > n-2: 
        return "Didnt converge"
    else: 
        return x_n

def newton_method(f, df, x_n, eps = 10**(-6), n = 1000):
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
    while np.linalg.norm(f_xn) > eps and n > 0: 
        sol = np.linalg.solve(df(x_n), -f_xn)
        x_n = x_n + sol
        print("x_n equals to: ", x_n)
        #np.linalg.lstsq can deal with non invertibel matrices
        #x_n = x_n + np.linalg.solve(df(x_n), -f_xn)
        f_xn = f(x_n)
        n = n - 1
    print("n equals: ",n)
    if np.linalg.norm(f_xn) < eps: 
        return x_n
    else: 
        return "Didnt converge."

def naive_optimization(f, dim, domain, eps_newton = 10**(-6), eps_derivative = 10**(-6),k =100, n = 1000):
    """Return a candidate for an optimum of f, if the procedure converges. 

    Args:
        f:              a function from \R^n to \R whose optimum we want to find
        dim:            dimension of the function
        domain:         an array A = [a,b] that defines the domain of the function (always a square) or None
        eps_newton:     sensitivity of the root finding process
        eps_derivative: sensitivity of the derivative approximation
        k:              number of gridpoints in each axis
        n:              maximum of iterations before stopping the procedure
       


    Returns:
        out: either an approximation for an optimum or a message if the procedure didnt converge

    """
    # 1. find point x_0 to start iteration from
        # For now we treat domain as the starting point of the iteration
    x_0 = np.array(find_starting_point(f,domain,20)).astype(float)

    # 2. compute derivative of f
    df = jacobian(f)
    # 3. compute jacobian of the derivative of f
    J = jacobian(df)
    # 4. run newton method on the derivative of f
    optimum = newton_method(df, J , x_0)
    # 5. return output of 4
    return(optimum)