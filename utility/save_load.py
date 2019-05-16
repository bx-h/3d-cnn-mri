#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   save_load.py
@Time    :   2019/04/14 21:00:15
@Author  :   Wu
@Version :   1.0
@Desc    :   None
'''

import torch

# import the files of mine
from logger import log


def save_model(model, path):
    """
    Save the model `model` to `path`\\
    Args:
        path: The path of the model to be saved
        model: The model to save
    """
    torch.save(model.state_dict(), path)
    log.logger.info("Model has been saved at: '{}'".format(path))


def load_model(model, path, device):
    """
    Load `model` from `path`, and push `model` to `device`\\
    Args:
        model: The model to save
        path: The path of the model to be saved
        device: the torch device
    Return:
        the loaded model
    """
    saved_params = torch.load(path)
    model.load_state_dict(saved_params)
    model = model.to(device)
    log.logger.info("Model has been loaded from: '{}'".format(path))
    return model
