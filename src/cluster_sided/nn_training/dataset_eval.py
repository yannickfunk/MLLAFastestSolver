import numpy as np
import torch
import random


class MatrixEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data_df):
        self.data_df = data_df.dropna().replace([True], 1).replace([False], 0)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        sample = self.data_df.iloc[index]
        solver_results = np.array(sample[1:8])
        solver_sorted = np.argsort(solver_results)

        features = sample[8:]

        # ['density', 'nonzeros', 'nsym', 'posdef', 'psym', 'rows']
        reduced = np.concatenate([features[4:5], features[10:15]])

        return np.array(reduced).astype("float64"), solver_sorted
