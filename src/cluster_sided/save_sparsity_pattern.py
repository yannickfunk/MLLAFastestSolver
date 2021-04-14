import os
import pandas as pd
from scipy.io import mmread
import numpy as np
import json
import matplotlib.pyplot as plt
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()

USER_PATH = os.environ["PROJECT"] + "/users/funk1/"
DATA_PATH = USER_PATH + 'data/'
SPARSITY_PATH = os.environ["PROJECT"] + "/data/funk1/sparsity/"
DATASET_PATH = DATA_PATH + 'dataset.csv'

def save_sparsity(sub_df):
    for _, row in sub_df.iterrows():
        label = row[1:8].astype("float64").idxmin()
        name = row["path"].split("/")[-1]
        mtx = mmread(row["path"])
        plt.spy(mtx, markersize=0.1)
        plt.axis("off")
        plt.savefig(SPARSITY_PATH+label+"/"+name+".png", bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"saved {SPARSITY_PATH+label+'/'+name+'.png'}")


df = pd.read_csv(DATASET_PATH).dropna()

if rank == 0:
    df = df.iloc[:340, :]
    save_sparsity(df)
    print("Worker 0 finished")
elif rank == 1:
    df = df.iloc[340:680, :]
    save_sparsity(df)
    print("Worker 1 finished")
elif rank == 2:
    df = df.iloc[680:1020, :]
    save_sparsity(df)
    print("Worker 2 finished")
elif rank == 3:
    df = df.iloc[1020:1360, :]
    save_sparsity(df)
    print("Worker 3 finished")
elif rank == 4:
    df = df.iloc[1360:1700, :]
    save_sparsity(df)
    print("Worker 4 finished")
elif rank == 5:
    df = df.iloc[1700:, :]
    save_sparsity(df)
    print("Worker 5 finished")


