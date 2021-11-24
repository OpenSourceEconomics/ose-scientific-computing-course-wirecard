import numpy as np
import numbers

# In this File we are implementing real valued functions to test the optimization algorithms we implement

# The first four are taken from our main source

def griewank(x, a = 200):
    """Compute the output of the x.size dimensional griewank function. 

    Args:
        x: a number, an array of numbers or a numpy array of numbers
        

    Returns:
        out: value of the x.size dimensional griewank function evaluated at x 

    """
    
    # This function has been tested and should work properly 

    input = np.array(x)
    sum = (1 / 200) * np.dot(input,input)
    prod = np.prod( np.cos( input / np.sqrt( np.arange(1,input.size + 1,1) ) ) )
    out = sum - prod + 2
    return out

def levi_no_13(x):
    """Compute the output of the x.size dimensional Levi No. 13 function. 

    Args:
        x: a number, an array of numbers or a numpy array of numbers
        

    Returns:
        out: value of the x.size dimensional Levi No. 13 function evaluated at x 

    """

    # This function still hast to be tested wether it does what its supposed to do

    input = np.array(x)

    if input.size > 1 : 
        out_1 = np.sin(np.sin(3 * np.pi * input[0]))
        out_2 = ( (input[-1] - 1) ** 2 ) * (1 + np.sin(np.sin(2 * np.pi * input[-1]))) 
        out_3 = np.dot(((input[:-1] - 1)**2), (1 + np.sin( np.sin(3 * np.pi * input[1:]) )))
        out = out_1 + out_2 + out_3 + 1
    else: 
        out = np.sin(np.sin(3 * np.pi * input)) + ( (input - 1) ** 2 ) * (1 + np.sin(np.sin(2 * np.pi * input))) + 1 

    return(out)


x_1 = np.array([1,4,np.sqrt(3)])
x_2 = 5
x_3 = np.array([1,2,3,4,5])

#print( np.prod( np.cos( x_1 / np.sqrt( np.arange(1,x_1.size +1,1) ) ) ) )

#print( griewank(x_2) )
print( levi_no_13(x_1) )
print( levi_no_13(x_2) )
