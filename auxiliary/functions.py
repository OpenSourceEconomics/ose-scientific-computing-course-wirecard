import autograd.numpy as np
import numbers

# In this File we are implementing real valued functions to test the optimization algorithms we implement
# The first four are taken from our main source
# the parameter 'domain' is not used yet. Might be integrated in the future
# the classes at the end of the file can be ignored for now. They might be integrate in the future to improve on the project


# the griewank function
def griewank(x, a=200, domain=[-100, 100]):
    """Compute the output of the x.size dimensional griewank function.

    Args:
        x:      a number, an array of numbers or a numpy array of numbers
        a:      a variable factor of the function
        domain: the domain of the function


    Returns:
        out: value of the x.size dimensional griewank function evaluated at x

    """

    # This function has been tested and should work properly

    # The global minimum of this function lies at (0,...,0) and equals to 1

    input = np.array(x)
    sum = (1 / 200) * np.dot(input, input)
    prod = np.prod(np.cos(input / np.sqrt(np.arange(1, input.size + 1, 1))))
    out = sum - prod + 2
    return out


# the rastrigin function
def rastrigin(x, domain=[-5.12, 5.12]):
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

    A = 10
    input = np.array(x)
    n = len(input)
    out = A * n + np.sum(input * input - A * np.cos(2 * np.pi * input)) + 1
    return out


# the rosenbrock function
def rosenbrock(x, domain=[-100, 100]):
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

    if input.size <= 1:
        return 0
    else:
        return (
            np.sum(100 * (input[1:] - input[:-1] ** 2) ** 2 + (1 - input[:-1]) ** 2) + 1
        )


# the levi no. 13 function
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

    if input.size > 1:
        out_1 = np.sin(np.sin(3 * np.pi * input[0]))
        out_2 = ((input[-1] - 1) ** 2) * (1 + np.sin(np.sin(2 * np.pi * input[-1])))
        out_3 = np.dot(
            ((input[:-1] - 1) ** 2), (1 + np.sin(np.sin(3 * np.pi * input[1:])))
        )
        out = out_1 + out_2 + out_3 + 1
    else:
        out = (
            np.sin(np.sin(3 * np.pi * input))
            + ((input - 1) ** 2) * (1 + np.sin(np.sin(2 * np.pi * input)))
            + 1
        )

    return out


# a function that at some point is going to be used for the classes
def is_x_in_domain(x, domain):
    """check wether x lies in the domain

    Args:
        x:      a vector of lenght n
        domain: an array of 2 dimensional vectors of lenght n


    Returns:
        out:    true if x is in the domain, else: flalse

    """
    if len(x) != len(domain):
        assert "Dimension of input value and domain are not the same"
    for x_i, i in zip(x, range(len(x))):
        if domain[i][0] < x_i and x_i < domain[i][1]:
            pass
        else:
            return False

    return True


# the rastrigin function as a class
class rastrigin_instance:
    def __init__(self, dimension, A=10, domain=[-5.12, 5.12]):
        self.A = A
        self.dim = dimension
        self.domain = np.array([domain for i in range(self.dim)])

    def value(self, y):
        assert (
            len(y) == self.dim
        ), "The dimension of the input and the function are different"
        # assert is_x_in_domain(y, self.domain), "The input value is not in the domain of the function"
        return rastrigin(y, self.A)


# the rosenbrock function as a class
class rosenbrock_instance:
    def __init__(self, dimension, domain=[-100, 100]):
        self.dim = dimension
        self.domain = np.array([domain for i in range(self.dim)])

    def value(self, x):
        assert (
            len(x) == self.dim
        ), "The dimension of the input is and the function are different"
        # assert is_x_in_domain(x, self.domain), "The input value is not in the domain of the function"
        return rosenbrock(x)


# the levo_no_13 function as a class
class levi_no_13_instance:
    def __init__(self, dimension, domain=[-10, 10]):
        self.dim = dimension
        self.domain = np.array([domain for i in range(self.dim)])

    def value(self, x):
        assert (
            len(x) == self.dim
        ), "The dimension of the input is and the function are different"
        # assert is_x_in_domain(x, self.domain), "The input value is not in the domain of the function"
        return levi_no_13(x)


# the griewank function as a class
class griewank_instance:
    def __init__(self, dimension, a=200, domain=[-100, 100]):
        self.a = 200
        self.dim = dimension
        self.domain = np.array([domain for i in range(self.dim)])

    def value(self, x):
        assert (
            len(x) == self.dim
        ), "The dimension of the input is and the function are different"
        # assert is_x_in_domain(x, self.domain), "The input value is not in the domain of the function"
        return griewank(x, self.a, self.domain)


if __name__ == "__main__":

    x_1 = np.array([1, 4, np.sqrt(3)])
    x_2 = 5
    x_3 = np.array([1, 2, 3, 4, 5])

    # print( np.prod( np.cos( x_1 / np.sqrt( np.arange(1,x_1.size +1,1) ) ) ) )

    # print( griewank(x_2) )
    print(levi_no_13(x_1))
    print(levi_no_13(x_2))

    List = [griewank, levi_no_13, rosenbrock, rastrigin]
    inputs = np.array(
        [
            [1, 2, 3],
            [5, 6, 7],
            [-8, 5, 7],
            [5, 2, -1],
            [0, 1, 0],
            [2, -4, -5],
            [19, 17, 23],
            [1, -1, 0],
        ]
    )
    A = []
    for function in List:
        A.append([function(x) for x in inputs])
    for a in A:
        print(a)
