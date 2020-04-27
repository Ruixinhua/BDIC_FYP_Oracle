# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Liu Dairui
# @Time          : 2020/4/28 1:56
# @Function      : A set of function help to deal with data
import configuration
import math
import os
import pickle
import torch
import tools
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from data_split import DataSplit


def random_data(dataset1, dataset2, labels=None, batch_size=16, batch_level="all", seed=42):
    """
    Random two paired dataset
    Args:
        dataset1: A list of data
        dataset2: A list of data
        labels: A list of labels corresponding to the dataset1 and dataset2
        batch_size: The size of each batch
        batch_level: The level of batch, random all or random by char type
        seed: random seed

    Returns: Random dataset1 and dataset2

    """
    np.random.seed(seed)
    indices = list(range(len(dataset1)))
    np.random.shuffle(indices)
    dataset1 = [dataset1[i] for i in indices]
    dataset2 = [dataset2[i] for i in indices]
    batch_num = math.ceil(len(dataset1) / batch_size)

    if batch_level == "all":
        dataset1 = [torch.tensor(dataset1[i * batch_size:(i + 1) * batch_size]) for i in range(batch_num)]
        dataset2 = [torch.tensor(dataset2[i * batch_size:(i + 1) * batch_size]) for i in range(batch_num)]
    if labels:
        labels = [labels[i] for i in indices]
        labels = [labels[i * batch_size:(i + 1) * batch_size] for i in range(batch_num)]
        return dataset1, dataset2, labels
    return dataset1, dataset2


def get_paired_data(char_types=None, test_char_num=100, batch_level="all", labeled=False, num_works=0, transform=None,
                    batch_data_path="dataset_batch.pkl", debug=False):
    """

    Args:
        char_types: the test character types
        test_char_num: the number of test characters
        batch_level: "all" get data by single image; "char" means get data with char level
        labeled: boolean type, if True will return labels list, default is Flase
        num_works: the number of workers
        transform: image transform
        batch_data_path: the batch data saved path
        debug: boolean type, if True will not load data from batch_data_path

    Returns: batch of dataset1, dataset2 or labels

    """
    """
    batch_level=
    output: dataset1 batch, dataset2 batch, or labels, character list
    """
    if char_types is None:
        char_types = ["jia", "jin"]
    dataset1_dir = tools.get_cur_dataset_path(char_types[0])
    dataset2_dir = tools.get_cur_dataset_path(char_types[1])
    if os.path.exists("char_list.pkl"):
        character_list = pickle.load(open("char_list.pkl", "rb"))
    else:
        character_sets = tools.get_char_set(dataset1_dir)
        if character_sets != tools.get_char_set(dataset2_dir):
            print("The two dataset are not identical")
        character_list = list(character_sets)
        pickle.dump(character_list, open("char_list.pkl", "wb"))

    print("load data by char")
    if os.path.exists(batch_data_path) and not debug:
        if labeled:
            dataset1_batch, dataset2_batch, labels = pickle.load(open(batch_data_path, "rb"))
            return dataset1_batch, dataset2_batch, labels, character_list
        else:
            dataset1_batch, dataset2_batch = pickle.load(open(batch_data_path, "rb"))
    else:
        dataset1_batch, dataset2_batch, labels = [], [], []
        for char in character_list[test_char_num:]:
            data1_loader = get_data_loader(os.path.join(dataset1_dir, char), num_works=num_works, transform=transform)
            data2_loader = get_data_loader(os.path.join(dataset2_dir, char), num_works=num_works, transform=transform)
            for batch1, batch2 in zip(data1_loader, data2_loader):
                if batch_level == "all":
                    dataset1_batch.extend(batch1[0].data.cpu().numpy())
                    dataset2_batch.extend(batch2[0].data.cpu().numpy())
                    if labeled:
                        labels.extend([char for _ in range(len(batch1[0]))])
                if batch_level == "char":
                    if batch1[0].shape[0] == 1:
                        batch1[0] = torch.cat((batch1[0], batch1[0]), 0)
                        batch2[0] = torch.cat((batch2[0], batch2[0]), 0)
                    dataset1_batch.append(batch1[0])
                    dataset2_batch.append(batch2[0])
                    if labeled:
                        labels.append([char for _ in range(len(batch1[0]))])
        if labeled:
            pickle.dump((dataset1_batch, dataset2_batch, labels), open(batch_data_path, "wb"))
            return dataset1_batch, dataset2_batch, labels
        else:
            pickle.dump((dataset1_batch, dataset2_batch), open(batch_data_path, "wb"))
    return dataset1_batch, dataset2_batch


def get_data_by_type(char_type, char_included=None, transform=None):
    """
    Get a set of DataLoader by character type
    Args:
        char_type: the type of character
        char_included: a list which includes the chars need
        transform: image transform

    Returns: a list of char DataLoader and a list of corresponding char label

    """
    cur_dataset_path = tools.get_cur_dataset_path(char_type)
    if char_included is None:
        # get all characters data if it is none
        char_included = tools.get_char_set(cur_dataset_path)
    char_data, labels = [], []
    if not transform:
        transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    for char in os.listdir(cur_dataset_path):
        if char not in char_included: continue
        char_data_loader = get_data_loader(os.path.join(cur_dataset_path, char), batch_size=512, transform=transform)
        for data in char_data_loader:
            if data[0].shape[0] == 1:
                data[0] = torch.cat((data[0], data[0]), 0)
            char_data.append(data[0])
            labels.append(char)
    return char_data, labels


def get_data_loader(data_dir=None, data_type="jia", train_test_split=1, train_val_split=0, batch_size=16,
                    transform=None, num_works=0):
    """
    Get train, val, and test DataLoader
    Args:
        data_dir: directory of dataset
        data_type: type of dataset, "jia", or "jin"
        train_test_split: the ratio of train test split
        train_val_split: the ratio of train val split
        batch_size: the size of batch
        transform: the transform of images
        num_works: the number of workers

    Returns: the DataLoader tuple objects needed(train or/and(+ val or/and test))

    """
    if not data_dir:
        data_dir = os.path.join(configuration.cur_data_dir, data_type)
    if not transform:
        transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    train_loader, val_loader, test_loader = DataSplit(ImageFolder(data_dir, transform=transform), train_test_split,
                                                      train_val_split).get_split(batch_size, num_workers=num_works)
    if train_test_split == 1 and train_val_split == 0:
        return train_loader
    elif train_val_split == 0:
        return train_loader, test_loader
    elif train_test_split == 1:
        return train_loader, val_loader
    else:
        return train_loader, val_loader, test_loader
