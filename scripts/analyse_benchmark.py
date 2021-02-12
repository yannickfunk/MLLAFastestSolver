import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BENCHMARK_RESULTS = json.load(open("../benchmark_results.json"))
# MEASUREMENT_POINTS = np.array([1, 2, 3, 4])  # milliseconds
MEASUREMENT_POINTS = np.linspace(1, 200, 10)


def normalize(fastest, x):
    return x / fastest


def print_performance_profile(performance_matrix):
    max = performance_matrix.max().max()
    x_values = np.linspace(1, max, num=500)
    y_lists = []
    for solver in performance_matrix:
        column = performance_matrix[solver]
        y_values = [np.count_nonzero(column <= x) for x in x_values]
        df = pd.DataFrame(data=y_values)
        df.name = solver
        y_lists.append(df)

    fig, ax = plt.subplots()
    ax.set_xlabel("maximum slowdown factor over fastest")
    ax.set_ylabel("Num of measurement points")
    for y_values in y_lists:
        ax.plot(x_values, np.array(y_values).flatten(), label=y_values.name, alpha=0.7)
    plt.title(performance_matrix.name)
    plt.legend(loc="lower right")
    plt.show()


def get_residual_matrix(benchmark_result):
    matrix_name = benchmark_result["filename"].split("/")[-1]
    residuals_per_solver = dict()
    for (solver_name, solver) in benchmark_result["solver"].items():
        if solver["completed"]:
            residuals = get_relative_residuals(solver_name, solver)
            indices_non_trunc = np.searchsorted(np.array(solver["iteration_timestamps"]), (MEASUREMENT_POINTS / 1000.0))
            indices = [i if i < len(residuals) else len(residuals) - 1 for i in indices_non_trunc]
            residuals_per_solver[solver_name] = np.array(residuals)[indices]
    performance_matrix = pd.DataFrame(data=residuals_per_solver, index=MEASUREMENT_POINTS)
    performance_matrix.name = matrix_name
    return performance_matrix.replace([np.inf], 1.0e+100).replace([np.nan, 0.0], 1.0e-16)


def get_relative_residuals(solver_name, solver):
    residuals = solver["true_residuals"][::2] if solver_name == "bicgstab" else solver["true_residuals"]
    # relative_residuals = np.array(residuals) / residuals[0]
    return residuals

def get_normalized_residual_matrix(residual_matrix):
    for index in residual_matrix.index:
        min_res = residual_matrix.loc[index].min()
        residual_matrix.loc[index] = residual_matrix.loc[index].map(lambda x: normalize(min_res, x))
    return residual_matrix


def get_res_sums_normalized(residual_matrix):
    sums = np.array([residual_matrix[e].sum() for e in residual_matrix.columns])
    df = pd.DataFrame([sums], columns=residual_matrix.columns)
    df.iloc[0] = df.iloc[0].map(lambda x: x / df.iloc[0].min())
    return df


def get_res_sums(residual_matrix):
    sums = np.array([residual_matrix[e].sum() for e in residual_matrix.columns])
    return pd.DataFrame([sums], columns=residual_matrix.columns)


def print_solver_histogram(matrices_res_sums_df):
    bins = np.linspace(1, 20, 50)
    matrices_res_sums_df.plot.hist(bins=bins, alpha=0.5)
    plt.show()
    for solver in matrices_res_sums_df.columns:
        ax = matrices_res_sums_df[solver].plot.hist(bins=bins, alpha=0.5, legend=True)
        plt.show()


if __name__ == '__main__':
    residual_matrices = [get_normalized_residual_matrix(get_residual_matrix(e)) for e in BENCHMARK_RESULTS]
    matrices_res_sums = [get_res_sums(e)
                         for e in residual_matrices if not e.empty]
    print(pd.concat(matrices_res_sums))
    matrices_res_sums_df = pd.concat(matrices_res_sums) / len(MEASUREMENT_POINTS)

    print_solver_histogram(matrices_res_sums_df)

