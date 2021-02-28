import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Matrix:
    def __init__(self, path, residuals):
        self.path = path
        self.residuals = residuals


class Benchmark:
    def __init__(self,  benchmark_path, measurement_points=np.linspace(0, 200, 10)):
        self.measurement_points = measurement_points
        self.benchmark_results = json.load(open(benchmark_path))
        spd_df = pd.read_csv("../../data/dataset.csv")
        spd = list(spd_df[spd_df.nsym == 1][spd_df.posdef][["path"]]["path"].values)
        self.residual_matrices = [
            # e for e in [Matrix(e["filename"], self._matrix(e)) for e in self.benchmark_results if e["filename"] in spd] if not e.residuals.empty
            e for e in [Matrix(e["filename"], self._matrix(e)) for e in self.benchmark_results] if not e.residuals.empty
        ]

    def _matrix(self, benchmark_result):
        matrix_name = benchmark_result["filename"].split("/")[-1]
        residuals_per_solver = dict()
        for (solver_name, solver) in benchmark_result["solver"].items():
            if solver["completed"] and solver_name in ["cg", "bicg", "fcg", "cgs", "gmres", "bicgstab", "idr"]:
                residuals = self._relative_residuals(solver_name, solver)
                solver_time = solver["apply"]["time"]
                # iteration_timestamps = np.array(solver["iteration_timestamps"])
                iteration_timestamps = np.linspace(solver_time / len(residuals), solver_time, len(residuals))
                indices_non_trunc = np.searchsorted(iteration_timestamps,
                                                    (self.measurement_points / 1000.0))
                indices = [i if i < len(residuals) else len(residuals) - 1 for i in indices_non_trunc]
                residuals_per_solver[solver_name] = np.array(residuals)[indices]
        performance_matrix = pd.DataFrame(data=residuals_per_solver, index=self.measurement_points)
        performance_matrix.name = matrix_name
        return performance_matrix.replace([np.inf], 1.0e+100).replace([np.nan, 0.0], 1.0e-16)

    @staticmethod
    def _normalized_residual_matrix(residual_matrix):
        for index in residual_matrix.index:
            min_res = residual_matrix.loc[index].min()
            residual_matrix.loc[index] = residual_matrix.loc[index].map(lambda x: x / min_res)
        return residual_matrix

    @staticmethod
    def _relative_residuals(solver_name, solver):
        residuals = solver["true_residuals"][::2] if solver_name == "bicgstab" else solver["true_residuals"]
        return residuals

    @staticmethod
    def _res_sums(residual_matrix):
        sums = np.array([residual_matrix[e].sum() for e in residual_matrix.columns])
        return pd.DataFrame([sums], columns=residual_matrix.columns)

    @staticmethod
    def limit_residual_column(column):
        indices = column.index
        limited_column = []
        min_v = column.iloc[0]
        for _, v in column.iteritems():
            if v < min_v:
                min_v = v
            else:
                v = min_v
            limited_column.append(v)
        return pd.Series(data=limited_column, index=indices)

    def get_label_df(self):
        residual_matrices = self.get_matrices_with_normalized_residuals()
        non_empty_indices = [i for i, e in enumerate(residual_matrices) if not e.empty]
        matrices_res_sums = [self._res_sums(e)
                             for e in residual_matrices if not e.empty]
        filepaths = np.array([e["filename"] for e in self.benchmark_results])[non_empty_indices]
        matrices_res_sums_df = pd.concat(matrices_res_sums) / len(self.measurement_points)
        matrices_res_sums_df.insert(0, "path", filepaths)
        return matrices_res_sums_df

    def get_matrices(self):
        residual_matrices = []
        for res_mtx in self.residual_matrices:
            for solver in res_mtx.residuals.columns:
                res_mtx.residuals[solver] = self.limit_residual_column(res_mtx.residuals[solver])
            residual_matrices.append(res_mtx)
        return residual_matrices

    def get_matrices_with_normalized_residuals(self):
        return [self._normalized_residual_matrix(e.residuals) for e in self.get_matrices()]

    def plot_solver_histogram(self):
        matrices_res_sums = [self._res_sums(e)
                             for e in self.get_matrices_with_normalized_residuals()]
        matrices_res_sums_df = pd.concat(matrices_res_sums) / len(self.measurement_points)
        bins = np.linspace(1, 20, 50)
        matrices_res_sums_df.plot.hist(bins=bins, alpha=0.5)
        plt.show()
        for solver in matrices_res_sums_df.columns:
            ax = matrices_res_sums_df[solver].plot.hist(bins=bins, alpha=0.5, legend=True)
            plt.show()

