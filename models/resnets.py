#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   resnets.py
@Time    :   2019/04/13 21:01:57
@Author  :   Wu
@Version :   1.0
@Desc    :   None
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ResidualBlock(nn.Module):
    '''
    Residual Block
    '''

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.left(x)
        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    Implementaion of ResNet, construct the ResNet layers with basic blocks (e.g. ResidualBlock)\\
    Args:
        block: the basic block, e.g. ResidualBolck or other block
        layers: a list with 4 positive numbers, e.g. [3, 4, 6, 3]
        num_classes: the number of classes
        img_in_channels: the channel of the images, 1 for gray level image
    """

    def __init__(self, block, layers=[3, 4, 6, 3], num_classes=3, img_in_channels=1):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # the initial layers
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=img_in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # global average pooling layer
        self.global_avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc_ = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        '''
        make a layer with num_blocks blocks.
        '''
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # add the first block with downsample
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels
        # add the (num_blocks - 1) blocks
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_(x)

        return x


def resnet18(pretrained=False, num_classes=3, img_in_channels=1):
    """
    Return:
        the 18 layers ResNet model
    """
    model = ResNet(block=ResidualBlock, layers=[2, 2, 2, 2], num_classes=num_classes, img_in_channels=img_in_channels)

    if pretrained:
        print("使用预训练模型")
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k,
                                    v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def resnet34(pretrained=False, num_classes=3, img_in_channels=1):
    """
    Return:
        the 34 layers ResNet model
    """
    model = ResNet(block=ResidualBlock, layers=[
        3, 4, 6, 3], num_classes=num_classes, img_in_channels=img_in_channels)
    if pretrained:
        print("使用预训练模型")
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


if __name__ == '__main__':
    print(resnet34())
