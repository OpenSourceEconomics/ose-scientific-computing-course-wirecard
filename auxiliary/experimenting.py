import pandas as pd
import numpy as np

from callable_algorithms import our_nelder_mead_method, our_newton_based_optimization
from functions import griewank

if __name__ == "__main__":
    optimum = [0, 0]
    df = pd.DataFrame([], columns=["computed result", "function evaluations"])

    print(df)
    a = our_nelder_mead_method(griewank, [1, 2], 1e-6, 1e-6, 200)
    b = our_nelder_mead_method(griewank, [-100, 103], 1e-6, 1e-6, 200)
    a_1 = a[0]
    a_2 = a[1]
    A = [a_1, a_2]
    print(A)
    # df = pd.DataFrame([[a_1,a_2],a], columns=["success", "function evaluations"])
    a_series = pd.Series(a, index=df.columns)
    b_series = pd.Series(b, index=df.columns)
    print(a_series)
    df = df.append(a_series, ignore_index=True)
    df = df.append(b_series, ignore_index=True)
    df["correct result"] = pd.Series([optimum] * len(df))
    # df["success"] = np.where(np.allclose(df["correct result"],df["computed result"]),1,0)
    g = np.array(df["correct result"])
    h = np.array(df["computed result"])
    # df["success"] = np.where(np.allclose(g,h),1,0)
    print(g)
    print(h)
    i = np.where(np.array_equal(g, h), 1, 0)
    print("i equals: ", i)
    # df['New Column']= np.where((np.array(df["computed result"]) > 2) & (np.array(df["correct result"]) < 30), True, False)
    print(df)
    # print(a[0])

    pass
