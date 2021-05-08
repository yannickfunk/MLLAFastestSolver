from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.io import mmread
import pandas as pd
import math
import os
import numpy as np
from compute_features import get_feature_vec, get_feature_sequence_vec
from tqdm import tqdm
import json
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
USER_PATH = os.environ["PROJECT"] + "/users/funk1/"
DATA_PATH = USER_PATH + 'data/'
DATASET_PATH = DATA_PATH + 'dataset.csv'
# DATASET_PATH = "../../data/dataset.csv"

N = 64


def chunks_n(mtx, n):
    mtx = csr_matrix(mtx)
    block_size = (mtx.shape[0] // int(math.sqrt(n)))
    result_chunks = []
    for i in range(0, mtx.shape[0], block_size):
        for j in range(0, mtx.shape[1], block_size):
            result_mtx = coo_matrix(mtx[i:i+block_size, j:j+block_size])
            if result_mtx.shape[0] != result_mtx.shape[1] or result_mtx.shape[0] != block_size or result_mtx.shape[1] != block_size:
                continue
            result_chunks.append(result_mtx)
    return result_chunks


def chunks_blocksize(mtx, block_size):
    mtx = csr_matrix(mtx)
    result_chunks = []
    for i in range(0, mtx.shape[0], block_size):
        for j in range(0, mtx.shape[1], block_size):
            result_mtx = coo_matrix(mtx[i:i + block_size, j:j + block_size])
            result_chunks.append((result_mtx, i, j))
    return result_chunks


def get_feature_sequence_dict(paths, worker):
    feature_dict = dict()
    for path in tqdm(paths, desc=f"Worker: {worker}"):
        mtx = mmread(path)
        blocks = chunks_blocksize(mtx, 4096)
        feature_vec = np.concatenate([get_feature_sequence_vec(block, i, j) for block, i, j in blocks], axis=1)
        feature_dict[path] = feature_vec.tolist()
    return feature_dict


def get_feature_dict(paths, worker):
    feature_dict = dict()
    for path in tqdm(paths, desc=f"Worker: {worker}"):
        mtx = mmread(path)
        blocks = chunks_n(mtx, N)[:N]
        feature_vec = np.concatenate([get_feature_vec(block) for block in blocks], axis=0)
        feature_dict[path] = feature_vec.tolist()
    return feature_dict


df = pd.read_csv(DATASET_PATH).dropna()
df = df[df["rows"] > 1024]
mtx_paths = list(df["path"].values)


if rank == 0:
    res_dict = get_feature_sequence_dict(mtx_paths[:470], 0)
    with open(DATA_PATH + 'blocking_vec0.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into blocking_vec0.json")
elif rank == 1:
    res_dict = get_feature_sequence_dict(mtx_paths[470:940], 1)
    with open(DATA_PATH + 'blocking_vec1.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into blocking_vec1.json")
elif rank == 2:
    res_dict = get_feature_sequence_dict(mtx_paths[940:1410], 2)
    with open(DATA_PATH + 'blocking_vec2.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into blocking_vec2.json")
elif rank == 3:
    res_dict = get_feature_sequence_dict(mtx_paths[1410:], 3)
    with open(DATA_PATH + 'blocking_vec3.json', 'w') as outfile:
        json.dump(res_dict, outfile, indent=4, sort_keys=True)
        print("Saved feature_vecs into blocking_vec3.json")