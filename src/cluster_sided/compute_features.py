import os
import json
from scipy.io import mmread
import numpy as np
import pandas as pd
from collections import defaultdict
import more_itertools as mit
from statistics import mean

USER_PATH = os.environ["PROJECT"] + "/users/funk1/"
META_PATH = USER_PATH+'data/meta.json'
PATHS_PATH = USER_PATH+'data/paths.json'


def get_paths_dict():
    with open(PATHS_PATH) as paths:
        paths_list = json.load(paths)
        paths_dict = dict()
        for item in paths_list:
            matrix_id = item["id"]
            matrix_path = item["path"]
            paths_dict[matrix_id] = matrix_path
    return paths_dict


def get_meta_dict():
    with open(META_PATH) as meta:
        meta_list = json.load(meta)
        meta_dict = dict()
        for item in meta_list:
            matrix_id = item["id"]
            include = ["rows", "nonzeros", "posdef", "psym", "nsym"]
            meta_dict[matrix_id] = {k: v for k, v in item.items() if k in include}
    return meta_dict


def get_feature_path_dict():
    return {k: {**{"path": path}, **get_meta_dict()[k]} for k, path in get_paths_dict().items()}


def nnz_per_row(mtx):
    return np.unique(mtx.nonzero()[0], return_counts=True)[1]


def chunks_per_row(mtx):
    chunk_dict = defaultdict(list)
    for x, y in zip(mtx.row, mtx.col):
        chunk_dict[x].append(y)
    chunks = []
    chunksizes = []
    for row in list(chunk_dict.values()):
        groups = [list(group) for group in mit.consecutive_groups(row)]
        chunksizes.extend([len(group) for group in groups])
        chunks.append(len(groups))
    return np.array(chunks), np.array(chunksizes)


def get_feature_sequence_vec(mtx, x_block, y_block):
    mean_x = 0 if len(mtx.row) == 0 else np.mean(mtx.row)
    std_x = 0 if len(mtx.row) == 0 else np.std(mtx.row)
    min_x = 0 if len(mtx.row) == 0 else np.min(mtx.row)
    max_x = 0 if len(mtx.row) == 0 else np.max(mtx.row)
    mean_y = 0 if len(mtx.col) == 0 else np.mean(mtx.col)
    std_y = 0 if len(mtx.col) == 0 else np.std(mtx.col)
    min_y = 0 if len(mtx.col) == 0 else np.min(mtx.col)
    max_y = 0 if len(mtx.col) == 0 else np.max(mtx.col)
    density = mtx.getnnz() / (mtx.shape[0] * mtx.shape[1])
    return np.expand_dims(np.array([
        x_block,
        y_block,
        mean_x,
        std_x,
        min_x,
        max_x,
        mean_y,
        std_y,
        min_y,
        max_y,
        density,
    ]), axis=1)


def get_feature_vec(mtx):
    rows = mtx.shape[0]
    nonzeros = mtx.nnz
    density = mtx.getnnz() / (mtx.shape[0] * mtx.shape[1])
    avg_nnz = mtx.getnnz() / mtx.shape[0]
    nnz_p_r = nnz_per_row(mtx)
    nnz_p_r = np.array([0]) if len(nnz_p_r) == 0 else nnz_p_r
    max_nnz = int(nnz_p_r.max())
    std_nnz = np.std(nnz_per_row(mtx))
    chunks, chunk_sizes = chunks_per_row(mtx)
    chunks = np.array([0]) if len(chunks) == 0 else chunks
    chunk_sizes = np.array([0]) if len(chunk_sizes) == 0 else chunk_sizes
    avg_row_block_count = np.mean(chunks)
    std_row_block_count = np.std(chunks)
    min_row_block_count = np.min(chunks)
    max_row_block_count = np.max(chunks)
    avg_row_block_size = np.mean(chunk_sizes)
    std_row_block_size = np.std(chunk_sizes)
    min_row_block_size = np.min(chunk_sizes)
    max_row_block_size = np.max(chunk_sizes)
    block_count = np.sum(chunks)
    return np.array([
        rows,
        nonzeros,
        density,
        avg_nnz,
        max_nnz,
        std_nnz,
        avg_row_block_count,
        std_row_block_count,
        min_row_block_count,
        max_row_block_count,
        avg_row_block_size,
        std_row_block_size,
        min_row_block_size,
        max_row_block_size,
        block_count,
    ], dtype=np.float64)

def get_feature_df():
    feature_dict = get_feature_path_dict()
    for key, meta in feature_dict.items():
        if meta["nonzeros"] > 10000000:
            continue
        print(f'reading matrix {meta["path"]}')
        mtx = mmread(meta["path"])
        density = mtx.getnnz() / (mtx.shape[0] * mtx.shape[1])
        feature_dict[key]["density"] = density
        feature_dict[key]["avg_nnz"] = mtx.getnnz() / mtx.shape[0]
        feature_dict[key]["max_nnz"] = int(nnz_per_row(mtx).max())
        feature_dict[key]["std_nnz"] = np.std(nnz_per_row(mtx))
        chunks, chunk_sizes = chunks_per_row(mtx)
        feature_dict[key]["avg_row_block_count"] = np.mean(chunks)
        feature_dict[key]["std_row_block_count"] = np.std(chunks)
        feature_dict[key]["min_row_block_count"] = np.min(chunks)
        feature_dict[key]["max_row_block_count"] = np.max(chunks)
        feature_dict[key]["avg_row_block_size"] = np.mean(chunk_sizes)
        feature_dict[key]["std_row_block_size"] = np.std(chunk_sizes)
        feature_dict[key]["min_row_block_size"] = np.min(chunk_sizes)
        feature_dict[key]["max_row_block_size"] = np.max(chunk_sizes)
        feature_dict[key]["block_count"] = np.sum(chunks)

    return pd.DataFrame(data=feature_dict).T
