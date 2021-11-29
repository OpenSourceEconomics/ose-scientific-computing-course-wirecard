import autograd.numpy as np
import numbers

# In this File we are implementing real valued functions to test the optimization algorithms we implement

# The first four are taken from our main source

def griewank(x, a = 200, domain = [-100,100]):
    """Compute the output of the x.size dimensional griewank function. 

    Args:
        x: a number, an array of numbers or a numpy array of numbers
        a: a variable factor of the function
        

    Returns:
        out: value of the x.size dimensional griewank function evaluated at x 

    """
    
    # This function has been tested and should work properly 

    # The global minimum of this function lies at (0,...,0) and equals to 1

    input = np.array(x)
    sum = (1 / 200) * np.dot(input,input)
    prod = np.prod( np.cos( input / np.sqrt( np.arange(1,input.size + 1,1) ) ) )
    out = sum - prod + 2
    return out

def rastrigin(x, A = 10, domain = [-5.12,5.12]):
    """Compute the output of the x.size dimensional rastrigin function. 

    Args:
        x: a number, an array of numbers or a numpy array of numbers all entries of x should lie within the default domain
        A: A factor that varies the shape of the function
        domain: the domain of the function

    Returns:
        out: value of the x.size dimensional rastrigi function evaluated at x 

    """
    # The global minimum lies at (0,...,0) and equals to 1

    # This function still hast to be tested wether it does what its supposed to do

    input = np.array(x)
    n = input.size
    out = A * n + np.sum( input * input - A * np.cos(2 * np.pi * input) ) + 1
    return out

def rosenbrock(x, domain = [-100,100]):
    """Compute the output of the x.size dimensional rosenbrock function. 

    Args:
        x: a number, an array of numbers or a numpy array of numbers all entries of x should lie within the default domain
        domain: the domain of the function

    Returns:
        out: value of the x.size dimensional rosenbrock function evaluated at x 

    """
    # This function still hast to be tested wether it does what its supposed to do

    # The global mimimum of this function lies at (1,..,1) and equals to 1 

    input = np.array(x)
    
    if input.size <= 1 : 
        return 0
    else: 
        return np.sum(100 * (input[1:] - input[:-1]**2 )**2 + (1 - input[:-1])**2) + 1

# for now i have excluded the domain of the following function
def levi_no_13(x):
    """Compute the output of the x.size dimensional Levi No. 13 function. 

    Args:
        x: a number, an array of numbers or a numpy array of numbers all entries of x should lie within the default domain
        domain: the domain of the function

    Returns:
        out: value of the x.size dimensional Levi No. 13 function evaluated at x 

    """

    # This function still hast to be tested wether it does what its supposed to do

    # The global minimum of this function lies at (1,...1) and equals to 1

    input = np.array(x)


    if input.size > 1 : 
        out_1 = np.sin(np.sin(3 * np.pi * input[0]))
        out_2 = ( (input[-1] - 1) ** 2 ) * (1 + np.sin(np.sin(2 * np.pi * input[-1]))) 
        out_3 = np.dot(((input[:-1] - 1)**2), (1 + np.sin( np.sin(3 * np.pi * input[1:]) )))
        out = out_1 + out_2 + out_3 + 1
    else: 
        out = np.sin(np.sin(3 * np.pi * input)) + ( (input - 1) ** 2 ) * (1 + np.sin(np.sin(2 * np.pi * input))) + 1 

    return(out)


if __name__ == "__main__":

    x_1 = np.array([1,4,np.sqrt(3)])
    x_2 = 5
    x_3 = np.array([1,2,3,4,5])

    #print( np.prod( np.cos( x_1 / np.sqrt( np.arange(1,x_1.size +1,1) ) ) ) )

    #print( griewank(x_2) )
    print( levi_no_13(x_1) )
    print( levi_no_13(x_2) )
