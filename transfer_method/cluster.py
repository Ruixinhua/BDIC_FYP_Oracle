import dimension_reduction
import model_output
import os
import plotly.express as px
import pandas as pd
import numpy as np
import tools
from sklearn.cluster import estimate_bandwidth, MeanShift
import configuration
new_dir = os.path.join(configuration.dataset_root_dir, "paired_jia_jin_sep-%s" % configuration.set_iter_no)
outlier_dir = os.path.join(configuration.dataset_root_dir, "paired_jia_jin_out-%s" % configuration.set_iter_no)
model_paths = (os.path.join(configuration.model_root_dir, "jia_ae_base_full.pkl"),
               os.path.join(configuration.model_root_dir, "jin_ae_base_full.pkl"))
model_type = "ae"
configuration.create_dirs(["output/", "plot/", "reduction/"])
output_file = os.path.join("output", "%s_iter-%d.pkl" % (model_type, configuration.set_iter_no))
reduction_file = os.path.join("reduction", "%s_iter-%d.csv" % (model_type, configuration.set_iter_no))


def data_plot(df_data, title, saved_path=None, x="x", y="y", color="type", size_max=20):
    if not saved_path:
        saved_path = os.path.join("plot", "%s.html" % title)
    fig = px.scatter(df_data, title=title, x=x, y=y, color=color, size_max=size_max, hover_data=["type", "label"])
    # fig.update_traces(marker=dict(line=dict(width=0.1)), selector=dict(mode='markers'))
    fig.write_html(saved_path)
    print("Saved the plot in %s" % saved_path)


def main():
    model_outputs = model_output.get_model_output(model_paths=model_paths,model_type=model_type,output_file=output_file)
    # get reduction results here which is a DataFrame{"label":[],"x":[],"y":[],"size":[],"path":[],"type":[]}
    model_reduction_results = dimension_reduction.get_reduction_result(model_outputs, reduction_file=reduction_file)
    images_data_filter = pd.DataFrame({"label": [], "x": [], "y": [], "size": [], "path": [], "type": []})
    cluster_centers_df = pd.DataFrame({"label": [], "x": [], "y": [], "type": [], "size": []})
    multi_cluster = {}
    for group in model_reduction_results.groupby(["label", "type"]):
        image_group = group[1].reset_index()
        label = image_group.iloc[0]["label"]
        # group images data by label
        group_xy = image_group.loc[:, ("x", "y")]
        group_xy = np.array(group_xy)
        # 带宽，也就是以某个点为核心时的搜索半径
        bandwidth = estimate_bandwidth(group_xy, quantile=0.8, n_samples=group_xy.shape[0])
        bandwidth = 0.0001 if bandwidth <= 0 else bandwidth
        # 设置均值偏移函数
        ms = MeanShift(bandwidth=bandwidth, max_iter=500)
        # 训练数据
        ms.fit(group_xy)
        cluster_centers = np.array(ms.cluster_centers_)
        distance2centers = {i: {} for i in range(len(cluster_centers))}
        for i, point, l in zip(range(len(ms.labels_)), group_xy, ms.labels_):
            distance2centers[l][i] = np.linalg.norm(point - cluster_centers[l])
        for key, distance2center in distance2centers.items():
            # calculate the maximum distance in theory
            percentile_distance = np.percentile(np.array(list(distance2center.values())), [75, 25])
            max_distance = percentile_distance[0] + (percentile_distance[0] - percentile_distance[1])
            keep_index, remove_index = [], []
            for i in distance2center.keys():
                if distance2center[i] < max_distance:
                    keep_index.append(i)
                else:
                    remove_index.append(i)

            if len(keep_index) > 0:
                center = "%s-%s" % (group[0][0], group[0][1])
                if center not in multi_cluster:
                    multi_cluster[center] = 0
                multi_cluster[center] += 1
                cluster = cluster_centers[key]
                new_cluster = {"x": cluster[0], "y": cluster[1], "size": 1, "label": group[0][0], "type": group[0][1]}
                cluster_centers_df = cluster_centers_df.append(pd.Series(new_cluster), ignore_index=True)
            # reset label
            label_sep = label + "-%i" % key
            image_group.loc[keep_index, "label"] = label_sep
            image_group.loc[keep_index, "new_path"] = image_group.loc[keep_index].path.apply(
                lambda p: os.path.join(new_dir, group[0][1], group[0][0], label_sep, p.split(os.sep)[-1]))
            image_group.loc[remove_index, "new_path"] = image_group.loc[remove_index].path.apply(
                lambda p: os.path.join(outlier_dir, group[0][1], group[0][0], label_sep, p.split(os.sep)[-1]))
            tools.copy_files(image_group.loc[remove_index, "path"], image_group.loc[remove_index, "new_path"])
            tools.copy_files(image_group.loc[keep_index, "path"], image_group.loc[keep_index, "new_path"])
            # remove the outlier points and keep the normal points
            images_data_filter = images_data_filter.append(image_group.iloc[keep_index], ignore_index=True)
    print("After filter:", images_data_filter.shape)
    print("Center shape:", cluster_centers_df.shape)
    data_plot(images_data_filter, "jia-jin-distribution")
    data_plot(cluster_centers_df, "jia-jin-centers")


if __name__ == "__main__":
    main()
