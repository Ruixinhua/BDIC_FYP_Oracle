# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Liu Dairui
# @Time          : 2020/4/28 1:48
# @Function      : Some useful function of run models

import torch
import torch.nn as nn
import loss_helper


def run_batch(model, batch, model_type="ae", train=True, loss_function=nn.MSELoss(reduction="sum")):
    """
    Put a batch of data into model
    Args:
        model: Model object
        batch: Batch data
        model_type: Type of model
        train: Boolean type, default is True
        loss_function: loss function of model

    Returns: The code and loss value after run the batch

    """
    if train:
        return run_model(model, batch, model_type, loss_function)
    else:
        with torch.no_grad():
            return run_model(model, batch, model_type, loss_function)


def run_model(model, batch, model_type="ae", loss_function=nn.MSELoss(reduction="sum"), img_size=96):
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
    if model_type == "ae":
        batch = batch.view(-1, img_size * img_size)
        output, code = model(batch)
        loss = loss_function(output, batch)
    else:
        output, code, mu, logvar = model(batch)
        loss = loss_helper.vae_loss(output, batch, mu, logvar)
    return code, loss
