import os
import pandas as pd
from scipy.io import mmread
import numpy as np
import json
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()

USER_PATH = os.environ["PROJECT"] + "/users/funk1/"
DATA_PATH = USER_PATH + 'data/'
DATASET_PATH = DATA_PATH + 'dataset.csv'


def get_feature_dict(df):
    feature_vecs = dict()
    for index in df.index:
        sample = df.loc[index]
        encoded_length = 4096
        print(f'reading: {sample["path"]}')
        coo = mmread(sample["path"])
        rows = sample["rows"]
        mtx = np.array([coo.row / rows, coo.col / rows, coo.data])
        encoded_vec = np.zeros(encoded_length)
        embed_mtx = np.random.rand(encoded_length, 3)
        for i, row in enumerate(embed_mtx):
            encoded_vec[i] = np.dot(row, mtx).sum()
        feature_vec = np.expand_dims(np.append(encoded_vec, np.array(sample[8:])), axis=0).astype("float64")
        feature_vecs[sample["path"]] = feature_vec.tolist()
    return feature_vecs

print("loading df...")
df = pd.read_csv(DATASET_PATH).dropna().replace([True], 1).replace([False], 0)
print("loaded!")

"""
if rank == 0:
    res_dict = get_feature_dict(df.iloc[:250, :])
    with open(DATA_PATH + 'feature_vecs0.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into feature_vecs0.json")
elif rank == 1:
    res_dict = get_feature_dict(df.iloc[250:500, :])
    with open(DATA_PATH + 'feature_vecs1.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into feature_vecs1.json")
elif rank == 2:
    res_dict = get_feature_dict(df.iloc[500:750, :])
    with open(DATA_PATH + 'feature_vecs2.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into feature_vecs2.json")
elif rank == 3:
    res_dict = get_feature_dict(df.iloc[750:1000, :])
    with open(DATA_PATH + 'feature_vecs3.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into feature_vecs3.json")
elif rank == 4:
    res_dict = get_feature_dict(df.iloc[1000:1250, :])
    with open(DATA_PATH + 'feature_vecs4.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into feature_vecs4.json")
elif rank == 5:
    res_dict = get_feature_dict(df.iloc[1250:1500, :])
    with open(DATA_PATH + 'feature_vecs5.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into feature_vecs5.json")
elif rank == 6:
    res_dict = get_feature_dict(df.iloc[1500:1750, :])
    with open(DATA_PATH + 'feature_vecs6.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into feature_vecs6.json")
elif rank == 7:
    res_dict = get_feature_dict(df.iloc[1750:, :])
    with open(DATA_PATH + 'feature_vecs7.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into feature_vecs7.json")
"""
if rank == 0:
    res_dict = get_feature_dict(df.iloc[1750:1900, :])
    with open(DATA_PATH + 'feature_vecs7.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into feature_vecs7.json")
elif rank == 1:
    res_dict = get_feature_dict(df.iloc[1900:2000, :])
    with open(DATA_PATH + 'feature_vecs8.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into feature_vecs8.json")
elif rank == 2:
    res_dict = get_feature_dict(df.iloc[2000:, :])
    with open(DATA_PATH + 'feature_vecs9.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into feature_vecs9.json")