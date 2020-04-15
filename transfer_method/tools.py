import torch
import configuration
import os
from torchvision import transforms
import torchvision
from data_split import DataSplit
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from models import AE
from models import ResNet_VAE
import pandas as pd
import shutil

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
epochs = 200
lr = 1e-3
img_size = 96
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
CNN_embed_dim = 256  # latent dim extracted by 2D CNN
char_list = None


def copy_files(source_paths, des_paths, print_log=False):
    """
    将源文件移到目标文件夹
    """
    for source_path, des_path in zip(source_paths, des_paths):
        if not os.path.exists(os.path.dirname(des_path)):
            os.makedirs(os.path.dirname(des_path))
        shutil.copyfile(source_path, des_path)
        if print_log:
            print("Copy file from %s to %s" % (source_path, des_path))


def get_device():
    return configuration.device


def get_default_model_class(model_type="ae"):
    if model_type == "ae":
        return AE(input_shape=img_size * img_size)
    elif model_type == "vae":
        return ResNet_VAE(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=0, CNN_embed_dim=CNN_embed_dim)


def get_default_transform(model_type):
    if model_type == "ae":
        return transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    elif model_type == "vae":
        return transforms.Compose([transforms.Resize([img_size, img_size]), transforms.ToTensor()])


def get_model_by_state(state_dic_path, model_class, device=get_device()):
    if os.path.exists(state_dic_path):
        model_class.load_state_dict(torch.load(state_dic_path, map_location=device))
    return model_class.to(device)


def get_model_opt(state_dic_path, model_class, learning_rate=1e-3, device=get_device()):
    model = get_model_by_state(state_dic_path, model_class, device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


def get_model_by_type(model_type, device=get_device(), train=True, model_paths=None, transform=None):
    if model_paths:
        jia_model_path, jin_model_path = model_paths
    else:
        if model_type == "ae":
            jia_model_path, jin_model_path = "model/jia_ae_base_full.pkl", "model/jin_ae_base_full.pkl"
            # jia_model_path, jin_model_path = "model/jia_ae_base.pkl", "model/jin_ae_base.pkl"
        elif model_type == "vae":
            jia_model_path, jin_model_path = "model/jia_vae_base_full.pkl", "model/jin_vae_base_full.pkl"
        else:
            # if not one of them, return ae model path
            jia_model_path, jin_model_path = "model/jia_ae_base_full.pkl", "model/jin_ae_base_full.pkl"
    jia_model, jia_optimizer = get_model_opt(jia_model_path, get_default_model_class(model_type), lr, device)
    jin_model, jin_optimizer = get_model_opt(jin_model_path, get_default_model_class(model_type), lr, device)
    if not transform:
        transform = get_default_transform(model_type)
    if train:
        jia_model.train()
        jin_model.train()
    return jia_model, jia_optimizer, jin_model, jin_optimizer, transform


def get_data_by_type(char_type, char_included=char_list, transform=None):
    cur_dataset_path = get_cur_dataset_path(char_type)
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
    if not data_dir:
        data_dir = os.path.join(configuration.cur_data_dir, data_type)
    if not transform:
        transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
    train_loader, val_loader, test_loader = DataSplit(dataset, train_test_split, train_val_split).get_split(batch_size,
                                                                                                            num_workers=num_works)
    if train_test_split == 1 and train_val_split == 0:
        return train_loader
    elif train_val_split == 0:
        return train_loader, test_loader
    elif train_test_split == 1:
        return train_loader, val_loader
    else:
        return train_loader, val_loader, test_loader


def get_paired_data(char_types=None, test_char_num=100, batch_level="all", labeled=False, num_works=0, transform=None):
    """
    batch_level="all" get data by single image; "char" means get data with char level
    output: dataset1 batch, dataset2 batch, or labels, character list
    """
    if char_types is None:
        char_types = ["jia", "jin"]
    dataset1_dir = get_cur_dataset_path(char_types[0])
    dataset2_dir = get_cur_dataset_path(char_types[1])
    if os.path.exists("char_list.pkl"):
        test_char_num, character_list = pickle.load(open("char_list.pkl", "rb"))
    else:
        character_sets = get_char_set(dataset1_dir)
        if character_sets != get_char_set(dataset2_dir):
            print("The two dataset are not identical")
        character_list = list(character_sets)
        pickle.dump((test_char_num, character_list), open("char_list.pkl", "wb"))

    print("load data by char")
    if os.path.exists("dataset_batch.pkl") or (os.path.exists("dataset_batch_labeled.pkl") and labeled):
        if labeled:
            dataset1_batch, dataset2_batch, labels = pickle.load(open("dataset_batch_labeled.pkl", "rb"))
            return dataset1_batch, dataset2_batch, labels, character_list
        else:
            dataset1_batch, dataset2_batch = pickle.load(open("dataset_batch.pkl", "rb"))
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
            pickle.dump((dataset1_batch, dataset2_batch, labels), open("dataset_batch_labeled.pkl", "wb"))
            return dataset1_batch, dataset2_batch, labels, character_list
        else:
            pickle.dump((dataset1_batch, dataset2_batch), open("dataset_batch.pkl", "wb"))
    return dataset1_batch, dataset2_batch, character_list


def get_cur_dataset_path(char_type):
    return os.path.join(configuration.cur_data_dir, char_type)


def get_char_set(dataset_path):
    return set(os.listdir(dataset_path))


def random_data(dataset1_batch, dataset2_batch, labels=None, batch_size=16, batch_level="all", seed=42):
    np.random.seed(seed)
    indices = list(range(len(dataset1_batch)))
    np.random.shuffle(indices)
    dataset1_batch = [dataset1_batch[i] for i in indices]
    dataset2_batch = [dataset2_batch[i] for i in indices]
    batch_num = math.ceil(len(dataset1_batch) / batch_size)

    if batch_level == "all":
        dataset1_batch = [torch.tensor(dataset1_batch[i * batch_size:(i + 1) * batch_size]) for i in range(batch_num)]
        dataset2_batch = [torch.tensor(dataset2_batch[i * batch_size:(i + 1) * batch_size]) for i in range(batch_num)]
    if labels:
        labels = [labels[i] for i in indices]
        labels = [labels[i * batch_size:(i + 1) * batch_size] for i in range(batch_num)]
        return dataset1_batch, dataset2_batch, labels
    return dataset1_batch, dataset2_batch


def loss_function(recon_x, x, mu, logvar):
    # MSE = F.mse_loss(recon_x, x, reduction='sum')
    MSE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def run_batch(model, batch, model_type="ae", train=True, mse=nn.MSELoss(reduction="sum")):
    """
    input: model, batch data and model type; If train, then train=True
    output: return the output of the model, the code, and the loss
    """
    if train:
        return run_model(model, batch, model_type, mse)
    else:
        with torch.no_grad():
            return run_model(model, batch, model_type, mse)


def run_model(model, batch, model_type="ae", mse=nn.MSELoss(reduction="sum")):
    if model_type == "ae":
        batch = batch.view(-1, img_size * img_size)
        output, code = model(batch)
        loss = mse(output, batch)
    else:
        output, code, mu, logvar = model(batch)
        loss = loss_function(output, batch, mu, logvar)
    return code, loss


def mmd_err(code_vectors1, code_vectors2):
    delta = code_vectors1 - code_vectors2
    return torch.sum(torch.mm(delta, torch.transpose(delta, 0, 1)))


def cal_dis_err(code_vectors1, code_vectors2, labels=None, train=True, criterion="mean"):
    if train:
        return get_dis_err(code_vectors1, code_vectors2, labels, criterion)
    else:
        with torch.no_grad():
            return get_dis_err(code_vectors1, code_vectors2, labels, criterion)


def get_dis_err(code_vectors1, code_vectors2, labels=None, criterion="mean", mse=nn.MSELoss(reduction="sum")):
    if labels:
        dis_err = torch.tensor(0, dtype=torch.float)
        index = 0
        code_labels = pd.DataFrame({"index": [i for i in range(len(labels))], "labels": labels})
        for _, group in code_labels.groupby("labels"):
            code1, code2 = code_vectors1[group["index"].to_numpy(), :], code_vectors2[group["index"].to_numpy(), :]
            if criterion == "mean":
                if not index:
                    dis_err = mse(code1, code2)
                else:
                    dis_err += mse(code1, code2)
            else:
                if not index:
                    dis_err = mmd_err(code1, code2)
                else:
                    dis_err += mmd_err(code1, code2)
            index += 1
    else:
        if criterion == "mean":
            dis_err = mse(code_vectors1, code_vectors2)
        else:
            dis_err = mmd_err(code_vectors1, code_vectors2)
    return dis_err
