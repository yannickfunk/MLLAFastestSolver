import os
import json
from scipy.io import mmread
import numpy as np
import pandas as pd

USER_PATH = os.environ["PROJECT"] + "/users/funk1/"
META_PATH = USER_PATH+'data/meta.json'
PATHS_PATH = USER_PATH+'data/paths.json'

def get_paths_dict():
    paths_list = json.load(open(PATHS_PATH))
    paths_dict = dict()
    for item in paths_list:
        matrix_id = item["id"]
        matrix_path = item["path"]
        paths_dict[matrix_id] = matrix_path
    return paths_dict


def get_meta_dict():
    meta_list = json.load(open(META_PATH))
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


def get_feature_df():
    feature_dict = get_feature_path_dict()
    for key, meta in feature_dict.items():
        mtx = mmread(meta["path"])
        density = mtx.getnnz() / (mtx.shape[0] * mtx.shape[1])
        feature_dict[key]["density"] = density
        feature_dict[key]["average_nnz"] = mtx.getnnz() / mtx.shape[0]
        feature_dict[key]["max_nnz"] = int(nnz_per_row(mtx).max())
        feature_dict[key]["std_nnz"] = np.std(nnz_per_row(mtx))

    return pd.DataFrame(data=feature_dict).T
