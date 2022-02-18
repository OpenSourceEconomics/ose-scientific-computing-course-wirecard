import pandas as pd
import numpy as np

from callable_algorithms import our_nelder_mead_method, our_newton_based_optimization
from functions import griewank

if __name__ == "__main__":
    df = pd.DataFrame([], columns=["success", "function evaluations"])
    # print(df)
    a = our_nelder_mead_method(griewank, [1, 2], 1e-6, 1e-6, 200)
    a_1 = a[0]
    a_2 = a[1]
    A = [a_1, a_2]
    print(A)
    # df = pd.DataFrame([[a_1,a_2],a], columns=["success", "function evaluations"])
    a_series = pd.Series(a, index=df.index)
    df.append(a_series, ignore_index=True)
    print(df)
    print(a[0])

    pass
