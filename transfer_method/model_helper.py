# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Liu Dairui
# @Time          : 2020/4/28 1:48
# @Function      : Some useful function of run models

import torch
import torch.nn as nn
import loss_helper
import tools


def run_batch(model, batch, model_type="AE", train=True, mse_loss=nn.MSELoss(reduction="sum"), num_iter=None,
              device=tools.get_device()):
    """
    Put a batch of data into models
    Args:
        model: Model object
        batch: Batch data
        model_type: Type of models
        train: Boolean type, default is True
        mse_loss: loss function of models
        device: cuda device, default is 0
        num_iter: the number of iteration which is running

    Returns: The code and loss value after run the batch

    """
    batch = batch.to(device)
    if train:
        model.train()
        return run_model(model, batch, model_type, mse_loss, num_iter=num_iter)
    else:
        model.eval()
        with torch.no_grad():
            return run_model(model, batch, model_type, mse_loss, num_iter=num_iter)


def run_model(model, batch, model_type="AE", mse_loss=nn.MSELoss(reduction="sum"), img_size=96, num_iter=None):
    """
    Run models with batch data
    Args:
        model: Model object
        batch: Batch data
        model_type: Type of target_model
        mse_loss: loss function of target_model
        img_size: the size of image
        num_iter: the number of iteration which is running

    Returns: The code and loss value after run the batch

    """
    if model_type == "AE":
        batch = batch.view(-1, img_size * img_size)
        output, code = model(batch)
        loss = mse_loss(output, batch)
    elif model_type == "VQVAE":
        output, code, vq_loss = model(batch)
        loss = vq_loss + mse_loss(batch, output)
    else:
        output, code, mu, logvar = model(batch)
        if num_iter is not None:
            loss = loss_helper.beta_vae_loss(output, batch, mu, logvar, loss_type='H', num_iter=num_iter)
        else:
            loss = loss_helper.vae_loss(output, batch, mu, logvar)
    return code, loss


def get_center(model, batch, model_type="AE", num_iter=None):
    # get the center of this batch
    code, _ = run_batch(model, batch, train=False, model_type=model_type, num_iter=num_iter)
    return torch.mean(code.cpu().data, dim=0).reshape((1, -1)).numpy()[0]


def predict(target_data, target_labels, source_data, source_labels, target_model, source_model=None, top_n=10,
            model_type="AE", criterion="mmd", mode="instance"):
    if source_model is None:
        source_model = target_model

    correct_count, index_sum = 0, 0
    correct_char = {}
    target_outputs = []
    from sklearn.neighbors import KNeighborsClassifier
    source_outputs = [get_center(source_model, batch, model_type=model_type) for batch in source_data]
    classifier = KNeighborsClassifier(n_neighbors=top_n)
    classifier.fit(source_outputs, source_labels)
    if mode == "class":
        target_outputs = [get_center(target_model, batch, model_type=model_type) for batch in target_data]
    else:
        labels = []
        for batch, label in zip(target_data, target_labels):
            output = run_batch(target_model, batch, train=False, model_type=model_type)[0].cpu().numpy()
            target_outputs.extend(output)
            labels.extend([label for _ in range(len(output))])
        target_labels = labels
    top_n_chars = classifier.kneighbors(target_outputs, return_distance=False)
    for top_n_char, target_label in zip(top_n_chars, target_labels):
        predicted_chars = [source_labels[i] for i in top_n_char]
        if target_label in predicted_chars:
            correct_count += 1
            correct_index = predicted_chars.index(target_label)  # jia_label的rank
            index_sum += correct_index  # 预测排名总和
            if correct_index not in correct_char:
                correct_char[correct_index] = []
            correct_char[correct_index].append(target_label)
    accuracy = correct_count / len(target_outputs)
    # sort prediction result
    correct_char = {k: v for k, v in sorted(correct_char.items(), key=lambda i: i[0])}
    chars = list()
    for c in correct_char.values():
        chars.extend(c)
    return {"Accuracy": accuracy, "Sum of index": index_sum, "Correct char": correct_char, "Chars": set(chars)}
