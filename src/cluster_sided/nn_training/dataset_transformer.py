import numpy as np
import torch
import random
from scipy.io import mmread
import math


class MatrixDatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, data_df):
        self.data_df = data_df.dropna().replace([True], 1).replace([False], 0)
        self.features = dict()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        sample = self.data_df.iloc[index]
        coo = mmread(sample["path"])
        rows = sample["rows"]
        feature_vec = np.array([coo.row / rows, coo.col / rows, coo.data]).T
        return feature_vec, np.argmin(np.array(sample[1:8]))