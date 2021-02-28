import os
from compute_features import get_feature_df
from analyse_benchmark import get_label_df

USER_PATH = os.environ["PROJECT"] + "/users/funk1/"
DATASET_PATH = USER_PATH+'data/dataset.csv'

label_df = get_label_df()
print("got label df")
feature_df = get_feature_df()
print("got feature df")

dataset = label_df.merge(feature_df, on="path")
print("merged dfs")

dataset.to_csv(DATASET_PATH, index=False)
print("saved dataset")
