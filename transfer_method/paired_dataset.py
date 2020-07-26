# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Liu Dairui
# @Time          : 2020/5/6 16:06
# @Function      : It defines the class of dataset
import numpy as np

from configuration import Configuration
import os
import tools
import torch
from torch.utils.data import DataLoader

from image_folder_with_paths import ImageFolderWithPaths


def get_dataset_by_path(dataset_path, transform, char_included=None):
    """
    Get a set of DataLoader by dataset path
    Args:
        dataset_path: the type of character
        transform: the transform of data
        char_included: a list which includes the chars need

    Returns: a list of char data, a list of corresponding char label, and paths

    """
    char_included = list(os.listdir(dataset_path)) if char_included is None else char_included
    data, labels, paths = [], [], []
    for char in char_included:
        # fetch all tensor at once
        char_data_loader = get_data_loader(os.path.join(dataset_path, char), transform, batch_size=512)
        for batch in char_data_loader:
            data.extend(batch[0].cpu().numpy())
            labels.extend([char for _ in range(len(batch[0]))])
            paths.extend(batch[2])
    return data, labels, paths


def get_data_loader(dataset_dir, transform, batch_size=1024):
    return DataLoader(ImageFolderWithPaths(dataset_dir, transform), batch_size=batch_size)


class PairedDataset:

    def __init__(self, run_in_server=False, conf=None, transform=None):
        if run_in_server:
            # dataset root path in server
            root_dir = "/home/dairui/data/datasets/"
        else:
            root_dir = "C:\\Users\\Rui\\Documents\\dataset"

        conf = Configuration() if conf is None else conf
        self.conf = conf
        self.paired_exp_dir = os.path.join(root_dir, "paired_%s_exp" % conf.dataset_type, "_".join(conf.paired_chars))
        self.paired_ori_dir = os.path.join(root_dir, "paired_%s_ori" % conf.dataset_type, "_".join(conf.paired_chars))
        self.unpaired_ori_dir = os.path.join(root_dir, "unpaired_%s_ori" % conf.dataset_type)
        self.paired_chars = conf.paired_chars
        self.char_list = list(os.listdir(os.path.join(self.paired_exp_dir, self.paired_chars[0])))
        self.target_name, self.source_name = conf.paired_chars[0], conf.paired_chars[1]
        self.target_dir = os.path.join(self.paired_ori_dir, self.target_name)
        self.transform = tools.get_default_transform(self.conf.model_type) if transform is None else transform

    def split_dataset(self, batch_size=128, train_chars=None, val_chars=None):
        self._split_dataset(batch_size, train_chars, val_chars)

    def _split_dataset(self, batch_size=128, train_chars=None, val_chars=None):
        split_num = round(len(self.char_list)*0.1)
        train_chars = self.char_list[split_num:] if train_chars is None else train_chars
        val_chars = self.char_list[:split_num] if val_chars is None else val_chars
        self.train_chars, self.val_chars, self.batch_size = train_chars, val_chars, batch_size
        # prepare training dataset
        self._get_paired_data(self.train_chars)
        # full data is used for prediction
        source_dir = os.path.join(self.paired_ori_dir, self.source_name)
        self.source_data_full, self.source_labels_full, _ = get_dataset_by_path(source_dir, self.transform, self.char_list)
        # prepare validation dataset
        self.target_val, self.labels_val, self.paths_val = get_dataset_by_path(self.target_dir, self.transform, self.val_chars)
        # full expand data is used for prediction
        source_exp_dir = os.path.join(self.unpaired_ori_dir, self.source_name)
        if os.path.exists(source_exp_dir):
            exp_chars = os.listdir(source_exp_dir)
            self.source_data_exp, self.source_labels_exp, _ = get_dataset_by_path(source_exp_dir, self.transform, exp_chars)
        else:
            self.source_data_exp, self.source_labels_exp = None, None
            tools.print_log("Fail to load expansion data!!!")

    def random_data(self, dataset1, dataset2, labels, seed=42):
        """
        Random two paired dataset
        Args:
            dataset1: A list of dataset with shape (len(dataset), ?)
            dataset2: A list of dataset with shape (len(dataset), ?)
            labels: A list of labels corresponding to the dataset1 and dataset2
            seed: random seed

        Returns: Random batch dataset batch1, batch2 and batch labels with shape (batch_num, batch_size, ?)

        """
        np.random.seed(seed)
        indices = list(range(len(dataset1)))
        np.random.shuffle(indices)
        dataset1 = [dataset1[i] for i in indices]
        dataset2 = [dataset2[i] for i in indices]
        labels = [labels[i] for i in indices]
        batch_num = round(len(dataset1) / self.batch_size)

        batch_data1 = [torch.tensor(dataset1[i * self.batch_size:(i + 1) * self.batch_size]) for i in range(batch_num)]
        batch_data2 = [torch.tensor(dataset2[i * self.batch_size:(i + 1) * self.batch_size]) for i in range(batch_num)]
        labels = [labels[i * self.batch_size:(i + 1) * self.batch_size] for i in range(batch_num)]
        return batch_data1, batch_data2, labels

    def _get_paired_data(self, char_included=None):
        """

        Args:
            char_included: a list which includes the chars need

        Returns: A list of target data, source data and labels

        """
        char_included = self.char_list if char_included is None else char_included
        self.target_dir = os.path.join(self.paired_exp_dir, self.paired_chars[0])
        self.source_dir = os.path.join(self.paired_exp_dir, self.paired_chars[1])
        self.target_data, self.source_data, self.labels = [], [], []
        for char in char_included:
            data1_loader = get_data_loader(os.path.join(self.target_dir, char), self.transform)
            data2_loader = get_data_loader(os.path.join(self.source_dir, char), self.transform, )
            for batch1, batch2 in zip(data1_loader, data2_loader):
                if batch1[0].shape[0] == 1:
                    batch1[0] = torch.cat((batch1[0], batch1[0]), 0)
                    batch2[0] = torch.cat((batch2[0], batch2[0]), 0)
                self.target_data.extend(batch1[0].cpu().numpy())
                self.source_data.extend(batch2[0].cpu().numpy())
                self.labels.extend([char for _ in range(len(batch1[0]))])


if __name__ == "__main__":
    tools.print_log("Start")
    dataset = PairedDataset(conf=Configuration(dataset_type="cluster"))
    tools.print_log("End")
