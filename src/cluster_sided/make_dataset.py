try:
    import os
    from compute_features import get_feature_df
    from benchmark import Benchmark

    USER_PATH = os.environ["PROJECT"] + "/users/funk1/"
    DATA_PATH = USER_PATH + 'data/'

    print("creating_benchmark object..")
    benchmark = Benchmark(DATA_PATH+"final_benchmark.json")
    print("created!")

    label_df = benchmark.get_label_df()
    print("got label df")
    feature_df = get_feature_df()
    print("got feature df")

    dataset = label_df.merge(feature_df, on="path")
    print("merged dfs")

    dataset.to_csv(DATA_PATH + "dataset.csv", index=False)
    print("saved dataset")
except Exception as e:
    print(e)
