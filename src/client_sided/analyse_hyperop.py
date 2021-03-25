import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../../data/hyperop.csv")

keys = list(df.keys())[:-1]

for key in keys:
    plt.scatter(df[key], df["loss"])
    plt.xlabel(key)
    plt.ylabel("loss")
    plt.ylim((0, 2))
    plt.show()