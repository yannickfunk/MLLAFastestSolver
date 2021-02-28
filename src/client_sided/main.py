from src.client_sided.benchmark import Benchmark
import matplotlib.pyplot as plt
import numpy as np

benchmark = Benchmark("../../data/preliminary_benchmark.json", measurement_points=np.linspace(0, 200, 10))

res_mtx = benchmark.plot_solver_histogram()

"""
count = 0
for mtx in res_mtx:
    count += 1
    fig, ax = plt.subplots()
    for column in mtx.residuals.columns:
        ax.plot(mtx.residuals.index, mtx.residuals[column], label=column)
    ax.legend()
    plt.yscale("log")
    plt.title(mtx.path)
    plt.savefig(f"../../plots/plot{count}.png")
"""
