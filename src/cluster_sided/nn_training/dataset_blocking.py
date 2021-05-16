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
        sample = self.data_df.iloc[index]
        block_features = np.nan_to_num(np.array(self.feature_dict[sample["path"]], dtype=np.float)).T
        # hard_features = np.array(sample[8:]).astype("float64")
        if block_features.shape[0] == 1:
            block_features = np.concatenate([block_features, block_features])

        return block_features, np.argmin(np.array(sample[1:8]))