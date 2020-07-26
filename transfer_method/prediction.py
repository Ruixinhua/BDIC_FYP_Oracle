# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Liu Dairui
# @Time          : 2020/7/25 20:18
# @Function      : prediction class
import math
import pandas as pd
import numpy as np
import torch

import model_helper
from sklearn.neighbors import KNeighborsClassifier

import tools
from configuration import Configuration
from paired_dataset import PairedDataset


class Prediction:

    def __init__(self, target_data, target_labels, source_data, source_labels, paths, top_n=10, model_type="AE",
                 mode="instance", batch_size=256, set_type="val"):
        self.target_data, self.target_labels = target_data, target_labels
        self.source_data, self.source_labels = source_data, source_labels
        self.batch_size, self.top_n, self.model_type, self.mode = batch_size, top_n, model_type, mode
        self.set_type, self.paths = set_type, paths

    def __get_output_df(self, model, data, labels, i):
        data = torch.tensor(data[i * self.batch_size:(i + 1) * self.batch_size])
        output, _ = model_helper.run_batch(model, data, model_type=self.model_type, train=False)
        output = [o for o in output.cpu().numpy()]
        labels = labels[i * self.batch_size:(i + 1) * self.batch_size]
        return {"output": output, "label": labels}

    def set_source(self, source_data, source_labels, set_type=None, mode=None):
        self.source_data, self.source_labels = source_data, source_labels
        self.set_type = set_type if set_type else self.set_type
        self.mode = mode if mode else self.mode

    def predict(self, target_model, source_model):
        target_outputs = pd.DataFrame({"output": [], "label": [], "path": []})
        for i in range(math.ceil(len(self.target_data) / self.batch_size)):
            output = self.__get_output_df(target_model, self.target_data, self.target_labels, i)
            paths = self.paths[i * self.batch_size:(i + 1) * self.batch_size]
            output["path"] = paths
            target_outputs = target_outputs.append(pd.DataFrame(output), ignore_index=True)
        source_outputs = pd.DataFrame({"output": [], "label": []})
        for i in range(math.ceil(len(self.source_data) / self.batch_size)):
            output = pd.DataFrame(self.__get_output_df(source_model, self.source_data, self.source_labels, i))
            source_outputs = source_outputs.append(output, ignore_index=True)
        source_centers, source_labels = [], []
        for label, group_df in source_outputs.groupby(["label"]):
            output = [o for o in group_df["output"]]
            source_centers.append(np.mean(output, axis=0))
            source_labels.append(label)
        classifier = KNeighborsClassifier(n_neighbors=self.top_n)
        classifier.fit(source_centers, source_labels)
        target_centers, target_labels, paths = [], [], []
        for label, group_df in target_outputs.groupby(["label"]):
            output = [o for o in group_df["output"]]
            if self.mode == "instance":
                target_centers.extend(output)
                target_labels.extend([label for _ in group_df["label"]])
                paths.extend([p for p in group_df["path"]])
            else:
                target_centers.append(np.mean(output, dim=0).reshape((1, -1))[0])
                target_centers.append(label)
        count, index_sum = 0, 0
        correct_char, correct_paths = {}, []
        top_n_chars = classifier.kneighbors(target_centers, return_distance=False)
        for top_n_char, target_label, path in zip(top_n_chars, target_labels, paths):
            predicted_chars = [source_labels[i] for i in top_n_char]
            if target_label in set(predicted_chars):
                count += 1
                correct_index = predicted_chars.index(target_label)  # jia_label的rank
                index_sum += correct_index  # 预测排名总和
                if correct_index not in correct_char:
                    correct_char[correct_index] = []
                correct_char[correct_index].append(target_label)
                correct_paths.append(path)
        accuracy = count / len(target_outputs)
        # sort prediction result
        correct_char = {k: v for k, v in sorted(correct_char.items(), key=lambda j: j[0])}
        chars = list()
        for c in correct_char.values():
            chars.extend(c)
        keys = ["accuracy", "index_sum", "correct", "chars"]
        values = [accuracy, index_sum, correct_char, chars]
        return {"%s_%s" % (self.set_type, k): v for k, v in zip(keys, values)}, correct_paths


if __name__ == "__main__":
    tools.print_log("Start")
    dataset = PairedDataset(conf=Configuration(dataset_type="cluster"))
    tools.print_log("End")
