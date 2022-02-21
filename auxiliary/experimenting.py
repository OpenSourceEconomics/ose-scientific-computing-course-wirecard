import pandas as pd
import numpy as np

from callable_algorithms import our_nelder_mead_method, our_newton_based_optimization
from functions import griewank

if __name__ == "__main__":
    optimum = [0, 0]
    df = pd.DataFrame([], columns=["computed_result", "function_evaluations", "time"])

    algo_name = "our_nelder_mead_method"
    t_func_name = "griewank"
    n = 2
    computational_budget = 200

    print(df)
    a = our_nelder_mead_method(griewank, [1, 2], 1e-6, 1e-6, computational_budget)
    b = our_nelder_mead_method(griewank, [-100, 103], 1e-6, 1e-6, computational_budget)
    a_1 = a[0]
    a_2 = a[1]
    A = [a_1, a_2]
    print(A)
    # df = pd.DataFrame([[a_1,a_2],a], columns=["success", "function evaluations"])
    a_series = pd.Series(a + [1], index=df.columns)
    b_series = pd.Series(b + [1], index=df.columns)
    print(a_series)
    df = df.append(a_series, ignore_index=True)
    df = df.append(b_series, ignore_index=True)
    df["correct_result"] = pd.Series([optimum] * len(df))

    print(df)

    # df["success"] = np.where(np.isclose(df["correct_result"],df["computed_result"]),1,0)
    g = np.array(df["correct_result"])
    h = np.array(df["computed_result"])
    # df["success"] = np.where(np.allclose(g,h),1,0)
    df["algorithm"] = pd.Series([algo_name] * n)
    df["test_function"] = pd.Series([t_func_name] * n)
    df["computational_budget"] = pd.Series([computational_budget] * n)
    df["sample_size"] = pd.Series([n] * n)
    df["success"] = df.apply(
        lambda row: np.allclose(row.correct_result, row.computed_result), axis=1
    )
    df["success"] = df.apply(lambda row: row.success * 1, axis=1)
    print(g)
    print(h)
    i = np.where(np.array_equal(g, h), 1, 0)
    print("i equals: ", i)
    # df['New Column']= np.where((np.array(df["computed result"]) > 2) & (np.array(df["correct result"]) < 30), True, False)
    print(df)
    print("about to test")
    # print(a[0])
    print("we test something: ", df.iloc[0]["time"])
    columns = [
        "algorithm",
        "test_function",
        "computational_budget",
        "sample_size",
        "success_rate",
        "average_time",
        "average_function_evaluations",
    ]
    df_new = pd.DataFrame([])
    df_new["algorithm"] = df.iloc[0]["algorithm"]
    df_new["test_function"] = df.iloc[0]["test_function"]
    df_new["computational_budget"] = df.iloc[0]["computational_budget"]
    df_new["sample_size"] = df.iloc[0]["sample_size"]
    df_new["success_rate"] = pd.Series([df["success"].mean()])
    df_new["average_time"] = pd.Series([df["time"].mean()])
    df_new["average_function_evaluations"] = pd.Series(
        [df["function_evaluations"].mean()]
    )

    print("the following is the data summary frame: \n")
    print(df_new)
    pass
