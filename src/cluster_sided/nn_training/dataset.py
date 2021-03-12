import numpy as np
import torch
import random


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, data_df, feature_vecs):
        self.data_df = data_df.dropna().replace([True], 1).replace([False], 0)
        self.feature_vecs = feature_vecs

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        sample = self.data_df.iloc[index]
        return np.array(self.feature_vecs[sample["path"]])[0], np.argmin(np.array(sample[1:8]))