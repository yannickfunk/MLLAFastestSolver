from compute_features import get_feature_df
from analyse_benchmark import get_label_df

dataset = get_label_df().merge(get_feature_df(), on="path")
dataset.to_csv("../dataset.csv", index=False)
