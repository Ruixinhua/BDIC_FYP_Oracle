# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Liu Dairui
# @Time          : 2020/7/22 12:00
# @Function      : The test of different paired dataset
import os

import torch

import tools
from configuration import ModelConfiguration
from paired_dataset import PairedDataset, get_dataset_by_path
from paired_trainer import PairedTrainer
import pandas as pd
import time

from prediction import Prediction

if __name__ == "__main__":
    # test code here, using ae models
    # model_types = ["VanillaVAE", "VQVAE"]
    model_type = "VanillaVAE"
    stages = ["add", "base"]
    test_num = 10
    results = pd.DataFrame(columns=["models type", "val accuracy", "val index sum", "test accuracy",
                                    "test index sum", "time usage", "stage", "dataset"])
    paired_chars_comb = [["jia", "jin"], ["jin", "chu"]]
    dataset_type = "cluster"
    tools.make_dir("output")  # output directory store the results of prediction, including predicted paths
    tools.make_dir("resume")  # resume directory store the file which used to resume test process
    tools.make_dir("result")  # result directory store the important information as csv file of prediction result
    for paired_chars in paired_chars_comb:
        for stage in stages:
            # for model_type in model_types:
            conf = ModelConfiguration(model_type=model_type, paired_chars=paired_chars, dataset_type=dataset_type,
                                      stage=stage, early_stop=20)
            dataset = PairedDataset(False, conf=conf)
            resume_path = "resume/resume_%s_%s.csv" % ("_".join(paired_chars), stage)
            if not os.path.exists(resume_path):
                char_list = dataset.char_list
                split_num = round(len(char_list) / test_num)
                char_split = [char_list[i*split_num:(i+1)*split_num] for i in range(test_num)]

                def split_char(n):
                    test, val = char_split[n // test_num], char_split[n % test_num]
                    if test == val:
                        return None
                    train = [c for c in char_list if c not in set(test) and c not in set(val)]
                    return {"test": ",".join(test), "val": ",".join(val), "train": ",".join(train)}
                char_lists_combination = [split_char(i) for i in range(test_num * test_num)]
                char_lists_combination = [c for c in char_lists_combination if c is not None]
                char_lists_combination = pd.DataFrame(char_lists_combination)
                char_lists_combination["checked"] = 0
                char_lists_combination.to_csv(resume_path)
            char_lists_combination = pd.read_csv(resume_path)
            i = 0
            for test_chars, group_df in char_lists_combination.groupby(["test"]):
                j = 0
                # prepare test dataset
                target_test, labels_test, paths_test = get_dataset_by_path(dataset.target_dir, dataset.transform,
                                                                           test_chars.split(","))
                for val_chars, train_chars in zip(group_df["val"], group_df["test"]):
                    prediction_test_results = []
                    cur_num = i*(test_num-1)+j
                    if char_lists_combination.loc[cur_num, 'checked'] > 0:
                        j += 1
                        continue
                    dataset.split_dataset(train_chars=train_chars.split(","), val_chars=val_chars.split(","))
                    conf.set_path(os.path.join("_".join(paired_chars), stage, str(cur_num)))
                    start_time = time.time()
                    test_trainer = PairedTrainer(config=conf, dataset=dataset)
                    test_trainer.train()
                    time_usage = time.time() - start_time
                    val_acc, val_index_sum = test_trainer.best_acc, test_trainer.best_index_sum
                    # predict on the test set
                    pred = Prediction(target_test, labels_test, dataset.source_data_full, dataset.source_labels_full,
                                      paths_test, model_type=model_type, set_type="test")
                    test_result, test_paths = pred.predict(test_trainer.target_model, test_trainer.source_model)
                    test_acc, test_index_sum = test_result["test_accuracy"], test_result["test_index_sum"]
                    prediction_test_results.append([test_result, test_paths])
                    # predict on the expand test set
                    pred.set_source(dataset.source_data_exp, dataset.source_labels_exp, set_type="test_exp")
                    test_result, test_paths = pred.predict(test_trainer.target_model, test_trainer.source_model)
                    prediction_test_results.append([test_result, test_paths])
                    tools.print_log("Val accuracy: %s, val index sum %s" % (val_acc, val_index_sum))
                    tools.print_log("Test accuracy: %s, test index sum %s" % (test_acc, test_index_sum))
                    result = {"models type": model_type, "val accuracy": val_acc, "val index sum": val_index_sum,
                              "test accuracy": test_acc, "test index sum": test_index_sum, "dataset": paired_chars,
                              "time usage": time_usage, "stage": stage}
                    results = results.append(pd.Series(result), ignore_index=True)
                    results.to_csv("result/results.csv")
                    char_lists_combination.loc[cur_num, "checked"] = 1
                    char_lists_combination.to_csv(resume_path)
                    torch.save(prediction_test_results, "output/%s_test.pkl" % cur_num)
                    torch.save(test_trainer.prediction_result, "output/%s_val.pkl" % cur_num)
                    j += 1
                i += 1

