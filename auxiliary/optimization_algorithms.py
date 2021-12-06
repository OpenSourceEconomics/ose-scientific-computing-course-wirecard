import autograd.numpy as np
from autograd import grad, jacobian


# In this File we are going to implement the optimization algorithms we study in our assignment

# To get some practice with implementing optimization algorithms I implemented the 1 Dimensional Newton Method, 
# and a naive optimization algorithms that uses the multidimensional newton method

def find_starting_point(f,domain,n,k = 10000):
    """Returns a candidate to start the local optimum finding process from.
    Args:
        f:              a function from \R^n to \R whose optimum we want to find
        domain:         the domain of the function in which we want to find the point (domain ist always a cube)
        n:              the dimension of the domain of the function
        k:              the amount of random points we draw to run the optimization on.

       


    Returns:
        out:            a candidate in domain^n to start the local search for an optimum from

    """
    #print("find_starting point bekommt n == ", n)
    A = np.random.rand(k,n)
    B = [f(domain[0] + (domain[1]-domain[0])*x ) for x in A]
    index = np.where(B == np.amin(B) )
    return A[index][0]

def initial_simplex(dim, domain):
    """Return a dim- dimensional simplex within the cube domain^n
    Args:
        dim:           the dimension we are working with

    Returns:
        out:           the verticies of the simplex in an dim+1 dimensional array

    """
    A = np.random.rand(dim + 1,dim)
    A = [domain[0] + x * (domain[1]- domain[0]) for x in A]
    
    return(A)




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

def nelder_mead_method(f, verts ,dim, alpha = 1, gamma = 2, rho = 0.5, sigma = 0.5):
    # Pseudo code can be found on: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

    # 0 Order

    values = np.array([f(vert) for vert,index in zip(verts,range(dim+1))])
    indexes = np.argsort(values)

    # 1 Termination

    if np.std(verts) < 10**(-6):
        return(verts[indexes[0]])

    # 2 Calculate x_0

    x_0 = np.sum(np.array([verts[i] for i in indexes[:-1]])) / dim

    # 3 Reflection

    x_r = x_0 + alpha*(x_0 - verts[indexes[-1]])
    if values[indexes[0]] <= f(x_r) and f(x_r) < values[indexes[-2]]:
        verts[indexes[-1]] = x_r
        return(nelder_mead_method(f, verts, dim , alpha, gamma, rho, sigma))

    # 4 Expansion

    if f(x_r) < values[indexes[0]]:
        x_e = x_0 + gamma*(x_r - x_0)
        if f(x_e) < f(x_r):
            verts[indexes[-1]] = x_e
            return(nelder_mead_method(f,verts,dim,alpha,gamma,rho,sigma))
        else: 
            verts[indexes[-1]] = x_r
            return(nelder_mead_method(f, verts, dim , alpha, gamma, rho, sigma))

    # 5 Contraction

    x_c = x_0 + rho * (verts[indexes[-1]] - x_0)
    if f(x_c) < f(verts[indexes[-1]]):
        verts[indexes[-1]] = x_c
        return(nelder_mead_method(f, verts, dim, alpha, gamma, rho, sigma))

    # 6 Shrink
    for i in range(indexes.size):
        if i != indexes[0]:
            verts[i] = verts[indexes[0]] + sigma*(verts[i] - verts[indexes[0]])
        return(nelder_mead_method(f, verts, dim, alpha, gamma, rho, sigma))
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
    test_initial_simplex = False
    if test_finding_starting_point == True: 
        print(find_starting_point(lambda a : a[0] + a[1], [4,6],2))
    if test_initial_simplex == True:
        simplex = initial_simplex(3,[5,6])
        for vert in simplex: 
            print(vert)