from analyse_benchmark import Benchmark
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

benchmark = Benchmark(measurement_points=np.linspace(0, 200, 200))


res_mtx = benchmark.get_residual_matrices()

count = 0
for mtx in res_mtx:
    count += 1
    fig, ax = plt.subplots()
    for column in mtx.data.columns:
        ax.plot(mtx.data.index, mtx.data[column], label=column)
    ax.legend()
    plt.yscale("log")
    plt.title(mtx.name)
    plt.savefig(f"../plots/plot{count}.png")

