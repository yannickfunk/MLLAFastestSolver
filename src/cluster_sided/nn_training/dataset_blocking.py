import numpy as np
import torch
import random
import json


class MatrixDatasetBlocking(torch.utils.data.Dataset):
    def __init__(self, data_df, feature_path):
        self.feature_dict = json.load(open(feature_path))
        data_df = data_df.dropna().replace([True], 1).replace([False], 0)
        self.data_df = data_df[data_df["path"].isin(self.feature_dict.keys())]

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        k = 1000
        sample = self.data_df.iloc[index]
        block_features = np.nan_to_num(np.array(self.feature_dict[sample["path"]], dtype=np.float)).T
        # hard_features = np.array(sample[8:]).astype("float64")

        padding = 0
        if block_features.shape[0] % k != 0:
            padding = (((block_features.shape[0] // k) * k) + k) - block_features.shape[0]
        padded = np.append(block_features, np.zeros((padding + k, block_features.shape[1])), axis=0)
        padded = padded.reshape(padded.shape[0] // k, block_features.shape[1] * k)
        print(padded.shape)
        return padded, np.argmin(np.array(sample[1:8]))