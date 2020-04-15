import os

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import configuration
from model_output import get_model_output


def get_reduction_result(model_outputs=None, reduction_file=configuration.model_reduction_path,
                         char_types=configuration.char_types):
    if not model_outputs:
        model_outputs = get_model_output()
    # 获取测试结果
    if os.path.exists(reduction_file):
        model_reduction_results = pd.read_csv(reduction_file)
        return model_reduction_results
    model_reduction_results = pd.DataFrame({"label": [], "x": [], "y": [], "size": [], "path": [], "type": []})
    index = 0
    for model_output in model_outputs:
        # reduce dimension to 2-D
        images_feature, images_label, images_path = model_output
        fea_dim = images_feature.shape[1]
        if fea_dim > 512:
            # 使用PCA降至512维
            pca = PCA(n_components=512)
            images_feature = pca.fit_transform(images_feature)
            print("Variance of pca", np.sum(pca.explained_variance_ratio_))
        # 使用TSNE降至2维
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=400)
        tsne_features_result = tsne.fit_transform(images_feature)
        print("Shape of features after reduce dimension", np.array(tsne_features_result).shape)
        for label, xy, path in zip(images_label, tsne_features_result, images_path):
            new_image = {"label": label, "x": xy[0], "y": xy[1], "size": 1, "path": path, "type": char_types[index]}
            model_reduction_results = model_reduction_results.append(pd.Series(new_image), ignore_index=True)
        index += 1
    model_reduction_results.to_csv(reduction_file, encoding="utf-8")
    return model_reduction_results

