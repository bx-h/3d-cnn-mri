#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# python main.py --model resnet --model_depth 34 --n_classes 3


import torch
import torch.utils.data as Data
import torch.nn as nn
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from PIL import Image

# import the files of mine
from logger import log, tensorboard_dir, is_aug, model_name, PATH_trained_model, PATH_log, PATH_patient_result, \
    PATH_tblogs
import utility.save_load
import utility.fitting
import process.load_dataset
# import models.resnets
import utility.evaluation
import random


from models.resnet import *
# from models.densenet import *

from opts import parse_opts
from model import generate_model


# remove the tensorboard_dir before running
try:
    os.system('rm -r {}'.format(tensorboard_dir))
except Exception as e:
    print(e)

# Device configuration, cpu, cuda:0/1/2/3 available
device = torch.device('cuda:0')

data_chooses = [2]  # choose dataset. 0: the small dataset, 1: CC_ROI, 2: 6_ROI
num_classes = 3

# Hyper parameters
batch_size = 32
num_epochs = 300
lr = 0.001
momentum = 0.9
weight_decay = 0.0001
is_WeightedRandomSampler = False
is_class_weighted_loss_func = True

# data processing
is_spacing = True
std_spacing_method = "global_std_spacing_mode"

# Log the preset parameters and hyper parameters
log.logger.info("Preset parameters:")
log.logger.info('model_name: {}'.format(model_name))
log.logger.info('data_chooses: {}'.format(data_chooses))
log.logger.info('num_classes: {}'.format(num_classes))
log.logger.info('device: {}'.format(device))

log.logger.info("Hyper parameters:")
log.logger.info('batch_size: {}'.format(batch_size))
log.logger.info('num_epochs: {}'.format(num_epochs))
log.logger.info('lr: {}'.format(lr))
log.logger.info('momentum: {}'.format(momentum))
log.logger.info('weight_decay: {}'.format(weight_decay))
log.logger.info('is_WeightedRandomSampler: {}'.format(is_WeightedRandomSampler))
log.logger.info('is_class_weighted_loss_func: {}'.format(is_class_weighted_loss_func))
log.logger.info('is_spacing: {}'.format(is_spacing))
log.logger.info('std_spacing_method: {}'.format(std_spacing_method))

# init datasets
mean_std, max_size_spc, global_hw_min_max_spc_world = process.load_dataset.init_dataset(
    data_chooses=data_chooses, test_size=0.2, std_spacing_method=std_spacing_method, new_init=True
)
log.logger.info('mean_std: {}'.format(mean_std))
log.logger.info('max_size_spc: {}'.format(max_size_spc))
log.logger.info('global_hw_min_max_spc_world: {}'.format(global_hw_min_max_spc_world))
# exit()
# data augmentation
# transforms.RandomRotation(degrees=[-10, 10]),
# transforms.RandomCrop(size=384)
# transforms.RandomHorizontalFlip(p=0.5)
train_transform = transforms.Compose([])

if is_aug:
    train_transform = transforms.Compose([
        # transforms.CenterCrop(size=max_size_spc),
        transforms.RandomRotation(degrees=[-10, 10]),
        # transforms.CenterCrop(size=512)
    ])

train_eval_transform = transforms.Compose([
    # transforms.CenterCrop(size=max_size_spc)
    # transforms.CenterCrop(size=384)
])

test_transform = transforms.Compose([
    # transforms.CenterCrop(size=max_size_spc)
    # transforms.CenterCrop(size=384)
])

# load datasets
train_data = process.load_dataset.MriDataset(train=True, transform=train_transform, is_spacing=is_spacing)
train_eval_data = process.load_dataset.MriDataset(train=True, transform=train_eval_transform, is_spacing=is_spacing)
test_data = process.load_dataset.MriDataset(train=False, transform=test_transform, is_spacing=is_spacing)


def checkImage(num=5):
    """
    在本地机器上运行，打开图片查看，检查，函数结束时会退出程序 (exit)
    """
    for _ in range(num):
        img_index = random.randint(1, 100)
        print(train_data[img_index][0].shape)
        print(train_data[img_index][0].dtype)
        print(train_data[img_index][0])
        np_img = train_data[img_index][0][0].numpy()
        pil_image = Image.fromarray(np_img)  # 数据格式为(h, w, c)
        print(pil_image)
        plt.imshow(np_img, cmap='gray')
        plt.show()

    exit()


# checkImage(5)

# get the class weight of train dataset, used for the loss function
class_weight = train_data.get_class_weight()
log.logger.info('class_weights: {}'.format(class_weight))

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False,
                                           sampler=train_data.get_sampler()) if is_WeightedRandomSampler else torch.utils.data.DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True)
train_loader_eval = torch.utils.data.DataLoader(dataset=train_eval_data, batch_size=batch_size,
                                                shuffle=False)  # train dataset loader without WeightedRandomSampler, for evaluation
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size,
                                          shuffle=False)  # test dataset loader, for evaluation

# Declare and define the model, optimizer and loss_func
# model = models.resnets.resnet18(pretrained=True, num_classes=num_classes, img_in_channels=1)

# model = resnet34(pretrained=False, num_classes=num_classes)
=======
model = resnet34(pretrained=True, num_classes=num_classes)
# model = resnet152(pretrained=True, num_classes=num_classes)
# model = densenet121(pretrained=True, num_classes=num_classes)

# ------  use other's 3d resnet ----------
# opt = parse_opts()
# model, parameters = generate_model(opt)

# ----- transform wrong ----------------------
# for name, module in model.named_modules():
#     if(isinstance(module, nn.Conv2d)):
#         kernel_size = module.kernel_size[0]
#         stride = module.stride[0]
#         padding = module.padding[0]
#         weight = module.weight.unsqueeze(2) / kernel_size
#         weight = torch.cat([weight for _ in range(0, kernel_size)], dim=2)
#         bias = module.bias
#
#         if(bias is None):
#             modules[name] = nn.Conv3d(in_channels=module.weight.shape[1], out_channels=module.weight.shape[0],
#                                kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
#         else:
#             module[name] = nn.Conv3d(in_channels=module.weight.shape[1], out_channels=module.weight.shape[0],
#                                kernel_size=kernel_size, padding=padding, stride=stride, bias=True)
#             module[name].bias = bias
#
#             module[name].weight.data = weight
#
#     elif(isinstance(module, nn.BatchNorm2d)):
#         weight = module.weight
#         bias = module.bias
#         module[name] = nn.BatchNorm3d(weight.shape[0])
#         module[name].weight = weight
#         module[name].bias = bias
#
# for name in module:
#     parent_module = model
#     objs = name.split(".")
#     if len(objs) == 1:
#         model.__setattr__(name, module[name])
#         continue
#
#     for obj in objs[:-1]:
#         parent_module = parent_module.__getattr__(obj)
#
#     parent_module.__setattr__(objs[-1], module[name])
#
#

optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
loss_func = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weight)) if is_class_weighted_loss_func else nn.CrossEntropyLoss()
log.logger.info(model)

try:
    log.logger.critical('Start training')
    utility.fitting.fit(model, num_epochs, optimizer, device, train_loader, test_loader, train_loader_eval, num_classes,
                        loss_func=loss_func, lr_decay_period=30, lr_decay_rate=2)
except KeyboardInterrupt as e:
    log.logger.error('KeyboardInterrupt: {}'.format(e))
except Exception as e:
    log.logger.error('Exception: {}'.format(e), exc_info=True) # 输出error的traceback
finally:
    log.logger.info("Train finished")
    utility.save_load.save_model(
        model=model,
        path='{}{}.pt'.format(PATH_trained_model, model_name)
    )
    model = utility.save_load.load_model(
        model=model,
        path='{}{}.pt'.format(PATH_trained_model, model_name),
        device=device
    )
    utility.evaluation.evaluate(model=model, val_loader=train_loader_eval, device=device, num_classes=3, test=False)
    utility.evaluation.evaluate(model=model, val_loader=test_loader, device=device, num_classes=3, test=True)
    log.logger.info('Finished')
