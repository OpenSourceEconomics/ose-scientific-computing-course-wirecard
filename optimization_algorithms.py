import autograd.numpy as np
from autograd import grad, jacobian


# In this File we are going to implement the optimization algorithms we study in our assignment

# To get some practice with implementing optimization algorithms I implemented the 1 Dimensional Newton Method, 
# and a naive optimization algorithms that uses the multidimensional newton method

def find_starting_point(f,domain,n,k = 10000):
    #print("find_starting point bekommt n == ", n)
    A = np.random.rand(k,n)
    B = [f(domain[0] + (domain[1]-domain[0])*x ) for x in A]
    index = np.where(B == np.amin(B) )
    return A[index][0]



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
    #print("newton method bekommt als x_n: ", x_n)
    f_xn = f(x_n)
    while np.linalg.norm(f_xn) > eps and n > 0: 
        sol = np.linalg.solve(df(x_n), -f_xn)
        x_n = x_n + sol
        #print("x_n equals to: ", x_n)
        #np.linalg.lstsq can deal with non invertibel matrices
        #x_n = x_n + np.linalg.solve(df(x_n), -f_xn)
        f_xn = f(x_n)
        n = n - 1
    #print("n equals: ",n)
    if np.linalg.norm(f_xn) < eps: 
        return x_n
    else: 
        return "Didnt converge."

def nelder_mead_method(f, dim, alpha = 1, gamma = 2, rho = 0.5, sigma = 0.5):
    # Pseudo code can be found on: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

    # 1 Order

    # 2 Calculate x_0

    # 3 Reflection

    # 4 Expansion

    # 5 Contraction

    # 6 Shrink

    pass

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
    
    if len(domain) > 2:
        x_0 = domain
        #print("x_0 = domain; x_0 = ", x_0)
    elif len(domain) == 2:
        x_0 = np.array(find_starting_point(f,domain,dim)).astype(float)
        #print("x_0 by find_starting_point; x_0 = ", x_0)
    else: 
        #print("domain ist: ",domain)
        print("domain ist nicht so wie sie sein sollte")
    # 2. compute derivative of f
    df = jacobian(f)
    # 3. compute jacobian of the derivative of f
    J = jacobian(df)
    # 4. run newton method on the derivative of f
    optimum = newton_method(df, J , x_0)
    # 5. return output of 4
    return(optimum)

if __name__ == "__main__":
    
    test_finding_starting_point = False
    if test_finding_starting_point == True: 
        print(find_starting_point(lambda a : a[0] + a[1], [4,6],2))