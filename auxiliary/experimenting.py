import pandas as pd
import numpy as np

from callable_algorithms import our_nelder_mead_method, our_newton_based_optimization
from functions import griewank

if __name__ == "__main__":
    df = pd.DataFrame([], ["success", "function evaluations"])
    # print(df)
    a = our_nelder_mead_method(griewank, [1, 2], 1e-6, 1e-6, 200)
    df = pd.DataFrame(a, ["success", "function evaluations"])
    a_series = pd.Series(a, index=df.index)
    df.append(a_series, ignore_index=True)
    print(df)
    print(a[0])

    pass
