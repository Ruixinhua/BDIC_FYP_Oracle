# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Liu Dairui
# @Time          : 2020/4/28 1:48
# @Function      : Some useful function of run models

import torch
import torch.nn as nn
import loss_helper
import heapq
import tools


def run_batch(model, batch, model_type="AE", train=True, loss_function=nn.MSELoss(reduction="sum"),
              device=tools.get_device()):
    """
    Put a batch of data into model
    Args:
        model: Model object
        batch: Batch data
        model_type: Type of model
        train: Boolean type, default is True
        loss_function: loss function of model
        device: cuda device, default is 0

    Returns: The code and loss value after run the batch

    """
    batch = batch.to(device)
    if train:
        return run_model(model, batch, model_type, loss_function)
    else:
        with torch.no_grad():
            return run_model(model, batch, model_type, loss_function)


def run_model(model, batch, model_type="AE", loss_function=nn.MSELoss(reduction="sum"), img_size=96):
    """
    Run model with batch data
    Args:
        model: Model object
        batch: Batch data
        model_type: Type of model
        loss_function: loss function of model
        img_size: the size of image

    Returns: The code and loss value after run the batch

    """
    if model_type == "AE":
        batch = batch.view(-1, img_size * img_size)
        output, code = model(batch)
        loss = loss_function(output, batch)
    else:
        output, code, mu, logvar = model(batch)
        loss = loss_helper.vae_loss(output, batch, mu, logvar)
    return code, loss


def get_center(model, batch, model_type="AE"):
    # get the center of this batch
    code, _ = run_batch(model, batch, train=False, model_type=model_type)
    return torch.mean(code.cpu().data, dim=0).reshape((1, -1))


def predict(target_data, target_labels, source_data, source_labels, target_model, source_model=None, top_n=10,
            model_type="AE", criterion="mmd"):
    if source_model is None:
        source_model = target_model
    target_centers = [get_center(target_model, batch, model_type=model_type) for batch in target_data]
    source_centers = [get_center(source_model, batch, model_type=model_type) for batch in source_data]
    correct_count, index_sum = 0, 0
    correct_char = {}
    for target_center, target_label in zip(target_centers, target_labels):
        distances_list = [loss_helper.cal_dis_err(target_center, source_center, train=False, criterion=criterion) for
                          source_center in source_centers]
        # calculate top n minimum distance characters
        top_n_char = map(distances_list.index, heapq.nsmallest(top_n, distances_list))
        predicted_chars = [source_labels[i] for i in top_n_char]
        if target_label in set(predicted_chars):
            correct_count += 1
            correct_index = predicted_chars.index(target_label)  # jia_label的rank
            index_sum += correct_index  # 预测排名总和
            if correct_index not in correct_char:
                correct_char[correct_index] = []
            correct_char[correct_index].append(target_label)
    accuracy = correct_count / len(target_centers) if len(target_centers) else 0
    return {"Accuracy": accuracy, "Sum of index": index_sum, "Correct char": correct_char}
