from model_output import get_model_output_df
import os
import pandas as pd
import numpy as np
from sklearn.cluster import estimate_bandwidth, MeanShift
import pickle


def get_cluster_output_df(input_df=None, output_file="cluster/vae_base_two_1.pkl", debug=False):
    """
    get cluster result and store the result
    Args:
        input_df: A pandas data frame with columns “label", "type", "feature"
        output_file: The file where the result is stored
        debug: If True, then it will not load the output_file; default is False

    Returns: A pandas data frame with columns “label", "type", "feature", "center", "size"

    """
    if input_df is None:
        input_df = get_model_output_df()
    # 获取测试结果
    if os.path.exists(output_file) and not debug:
        output_df = pickle.load(open(output_file, "rb"))
        return output_df
    columns = input_df.columns
    output_df = pd.DataFrame(columns=columns)
    for (label, char_type), group_df in input_df.groupby(["label", "type"]):
        group_df = group_df.reset_index()
        # group images data by label
        feature = np.array(group_df.loc[:, "feature"])
        feature = np.stack(feature)
        # 带宽，也就是以某个点为核心时的搜索半径
        bandwidth = estimate_bandwidth(feature, quantile=0.8, n_samples=feature.shape[0])
        bandwidth = 0.0001 if bandwidth <= 0 else bandwidth
        # 设置均值偏移函数
        ms = MeanShift(bandwidth=bandwidth, max_iter=10000, n_jobs=-1)
        # 训练数据
        ms.fit(feature)
        cluster_centers = np.array(ms.cluster_centers_)
        group_df["center"] = list(ms.labels_)
        group_df["size"] = 1
        output_df = output_df.append(group_df, ignore_index=True)
        centers_df = pd.DataFrame({"feature": list(cluster_centers), "center": list(range(len(cluster_centers)))})
        centers_df["size"] = centers_df.center.apply(lambda p: sum(np.array(ms.labels_) == p))
        centers_df["label"], centers_df["type"] = label, char_type
        output_df = output_df.append(centers_df, ignore_index=True)
    print("shape of output after cluster", output_df.shape)
    pickle.dump(output_df, open(output_file, "wb"))
    return output_df


if __name__ == "__main__":
    # test code here
    from dimension_reduction import get_reduction_result

    test_df = get_cluster_output_df(
        get_reduction_result(get_model_output_df(output_file="output/vae_base_two_1.pkl", debug=True),
                             output_file="reduction/vae_base_two_1.pkl", debug=True))
