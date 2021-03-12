import os
import pandas as pd
from scipy.io import mmread
import numpy as np
import json
from mpi4py import MPI
from sklearn.decomposition import TruncatedSVD
rank = MPI.COMM_WORLD.Get_rank()

USER_PATH = os.environ["PROJECT"] + "/users/funk1/"
DATA_PATH = USER_PATH + 'data/'
DATASET_PATH = DATA_PATH + 'dataset.csv'


def get_feature_dict(df):
    n_components = 2048

    feature_vecs = dict()
    for index in df.index:
        sample = df.loc[index]
        print(f'reading: {sample["path"]}')
        coo = mmread(sample["path"])
        rows = sample["rows"]
        if rows <= n_components:
            svd = TruncatedSVD(n_components=rows-1)
            svd.fit(coo)
            encoded_vec = np.append(svd.explained_variance_ratio_, np.zeros(n_components - (rows - 1)))
        else:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            svd.fit(coo)
            encoded_vec = svd.explained_variance_ratio_
        feature_vec = np.expand_dims(np.append(encoded_vec, np.array(sample[8:])), axis=0).astype("float64")
        print(feature_vec.shape)
        feature_vecs[sample["path"]] = feature_vec.tolist()
    return feature_vecs

print("loading df...")
df = pd.read_csv(DATASET_PATH).dropna().replace([True], 1).replace([False], 0)
print("loaded!")


if rank == 0:
    res_dict = get_feature_dict(df.iloc[:250, :])
    with open(DATA_PATH + 'feature_svd0.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into feature_svd0.json")
elif rank == 1:
    res_dict = get_feature_dict(df.iloc[250:500, :])
    with open(DATA_PATH + 'feature_svd1.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into feature_svd1.json")
elif rank == 2:
    res_dict = get_feature_dict(df.iloc[500:750, :])
    with open(DATA_PATH + 'feature_svd2.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into feature_svd2.json")
elif rank == 3:
    res_dict = get_feature_dict(df.iloc[750:1000, :])
    with open(DATA_PATH + 'feature_svd3.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into feature_svd3.json")
elif rank == 4:
    res_dict = get_feature_dict(df.iloc[1000:1250, :])
    with open(DATA_PATH + 'feature_svd4.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into feature_svd4.json")
elif rank == 5:
    res_dict = get_feature_dict(df.iloc[1250:1500, :])
    with open(DATA_PATH + 'feature_svd5.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into feature_svd5.json")
elif rank == 6:
    res_dict = get_feature_dict(df.iloc[1500:1750, :])
    with open(DATA_PATH + 'feature_svd6.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into feature_svd6.json")
elif rank == 7:
    res_dict = get_feature_dict(df.iloc[1750:, :])
    with open(DATA_PATH + 'feature_vecs7.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into feature_svd7.json")