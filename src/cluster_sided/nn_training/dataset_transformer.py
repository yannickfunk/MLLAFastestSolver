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
        k = 1000
        sample = self.data_df.iloc[index]
        coo = mmread(sample["path"])
        rows = sample["rows"]
        nonzeros = len(coo.row)
        padding = 0
        if nonzeros % k != 0:
            padding = (((nonzeros // k) * k) + k) - nonzeros

        feature_vec = np.array([coo.row / rows, coo.col / rows, coo.data]).T
        padded = np.append(feature_vec, np.zeros((padding, 3)), axis=0)
        padded = padded.reshape(padded.shape[0] // k, 3 * k)
        print(padded.shape)
        return padded, np.argmin(np.array(sample[1:8]))