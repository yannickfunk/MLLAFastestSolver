import numpy as np
import torch
import random
from scipy.io import mmread


class MatrixDatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, data_df):
        self.data_df = data_df.dropna().replace([True], 1).replace([False], 0)
        self.features = dict()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        sample = self.data_df.iloc[index]

        if index in list(self.features.keys()):
            mtx = self.features[index]
        else:
            coo = mmread(sample["path"])
            rows = sample["rows"]
            mtx = np.expand_dims(np.array([coo.row / rows, coo.col / rows, coo.data]).T.flatten(), axis=2)
            self.features[index] = mtx
            print(mtx.shape)
        return mtx, np.argmin(np.array(sample[1:8]))