# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Liu Dairui & Zhang Gechuan
# @Time          : 2020/4/28 1:43
# @Function      : The helper for loss calculation
import torch
import pandas as pd
import torch.nn as nn
from mmd import MMD_loss
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    # MSE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def mmd_err(code_vectors1, code_vectors2, kernel_type="linear"):
    return MMD_loss(kernel_type=kernel_type).forward(code_vectors1, code_vectors2)


def cal_dis_err(code_vectors1, code_vectors2, labels=None, train=True, criterion="mmd"):
    if train:
        return get_dis_err(code_vectors1, code_vectors2, labels, criterion)
    else:
        with torch.no_grad():
            return get_dis_err(code_vectors1, code_vectors2, labels, criterion)


def get_dis_err(code_vectors1, code_vectors2, labels=None, criterion="mmd", mse=nn.MSELoss(reduction="sum")):
    if labels:
        dis_err = torch.tensor(0, dtype=torch.float)
        index = 0
        labels_df = pd.DataFrame({"index": [i for i in range(len(labels))], "labels": labels})
        for _, group in labels_df.groupby("labels"):
            code1, code2 = code_vectors1[group["index"].to_numpy(), :], code_vectors2[group["index"].to_numpy(), :]
            if criterion == "mean":
                if not index:
                    dis_err = mse(code1, code2)
                else:
                    dis_err += mse(code1, code2)
            else:
                kernel_type = "linear"
                if criterion == "mmd-rbf":
                    kernel_type = "rbf"
                if not index:
                    dis_err = mmd_err(code1, code2, kernel_type)
                else:
                    dis_err += mmd_err(code1, code2, kernel_type)
            index += 1
    else:
        if criterion == "mean":
            dis_err = mse(code_vectors1, code_vectors2)
        else:
            kernel_type = "linear"
            if criterion == "mmd-rbf":
                kernel_type = "rbf"
            dis_err = mmd_err(code_vectors1, code_vectors2, kernel_type)
    return dis_err

