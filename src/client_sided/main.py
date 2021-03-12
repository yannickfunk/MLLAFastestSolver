from src.cluster_sided.benchmark import Benchmark
import numpy as np

benchmark = Benchmark("../../data/final_benchmark.json")

df = benchmark.get_label_df()
print(df[df["path"] == "/p/project/haf/data/funk1/.config/ssget/MM/Mycielski/mycielskian20/mycielskian20.mtx"].iloc[0])
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
