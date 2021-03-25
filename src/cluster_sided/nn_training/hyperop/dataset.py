import numpy as np
import torch
import random


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, data_df):
        self.data_df = data_df.dropna().replace([True], 1).replace([False], 0)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        sample = self.data_df.iloc[index]
        return np.array(sample[8:]).astype("float64"), np.argmin(np.array(sample[1:8]))