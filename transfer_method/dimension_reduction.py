import os

import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from model_output import get_model_output_df


def get_reduction_result(input_df=None, output_file="reduction/vae_base_two_1.pkl", debug=False):
    """
        get reduction result from input DataFrame object

    Args:
        input_df: A pandas data frame with columns “label", "type", "feature"
        output_file: The file where the result is stored
        debug: If True, then it will not load the output_file; default is False

    Returns:  pandas data frame with columns “label", "type", "feature" and "feature" column is 2-D.

    """
    if input_df is None:
        input_df = get_model_output_df(debug=True)
    if os.path.exists(output_file) and not debug:
        output_df = pickle.load(open(output_file, "rb"))
        return output_df
    columns = input_df.columns
    output_df = pd.DataFrame(columns=columns)
    for input_ in input_df.groupby(["type"]):
        char_type, output_ = input_
        # reduce dimension to 2-D
        feature = output_["feature"].values
        feature = np.stack(feature)
        fea_dim = feature[0].shape[0]
        if fea_dim > 512:
            # 使用PCA降至512维
            pca = PCA(n_components=512)
            feature = pca.fit_transform(feature)
            print("Variance of pca", np.sum(pca.explained_variance_ratio_))
        # 使用TSNE降至2维
        feature_reduction = TSNE(n_components=2, n_iter=12000, random_state=42).fit_transform(feature)
        # feature_reduction = PCA(n_components=2).fit_transform(feature)
        output_ = output_.drop(columns=["feature"])
        output_["feature"] = list(feature_reduction)
        feature_reduction = np.array(feature_reduction)
        print("Shape of features after reduce dimension", feature_reduction.shape)
        output_df = output_df.append(output_, ignore_index=True)
    pickle.dump(output_df, open(output_file, "wb"))
    return output_df


if __name__ == "__main__":
    # test code
    reduction_result = get_reduction_result(get_model_output_df(output_file="output/vae_base_two_1.pkl", debug=True),
                                            output_file="reduction/vae_base_two_1.pkl", debug=True)
