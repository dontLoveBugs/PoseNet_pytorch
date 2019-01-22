# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/21 19:51
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import torch.nn as nn
from torchvision import transforms, models, datasets
import torch.nn.functional as F


def model_parser(model, fixed_weight=False, dropout_rate=0.0, bayesian=False):

    if model == 'Googlenet':
        base_model = models.inception_v3(pretrained=True)
        network = GoogleNet(base_model, fixed_weight, dropout_rate)
    elif model == 'Resnet':
        base_model = models.resnet34(pretrained=True)
        network = ResNet(base_model, fixed_weight, dropout_rate, bayesian)
    elif model == 'ResnetSimple':
        base_model = models.resnet34(pretrained=True)
        network = ResNetSimple(base_model, fixed_weight)
    else:
        assert 'Unvalid Model'

    return network


class ResNet(nn.Module):
    def __init__(self, base_model, fixed_weight=False, dropout_rate=0.0, bayesian=False):
        super(ResNet, self).__init__()

        self.bayesian = bayesian
        self.dropout_rate = dropout_rate
        feat_in = base_model.fc.in_features

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.fc_last = nn.Linear(feat_in, 2048, bias=True)
        self.fc_position = nn.Linear(2048, 3, bias=True)
        self.fc_rotation = nn.Linear(2048, 4, bias=True)

        init_modules = [self.fc_last, self.fc_position, self.fc_rotation]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x_fully = self.fc_last(x)
        x = F.relu(x_fully)

        dropout_on = self.training or self.bayesian
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=dropout_on)

        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation, x_fully


class ResNetSimple(nn.Module):
    def __init__(self, base_model, fixed_weight=False, dropout_rate=0.0):
        super(ResNetSimple, self).__init__()
        self.dropout_rate = dropout_rate
        feat_in = base_model.fc.in_features

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        """
            the difference from ResNet
        """
        # self.fc_last = nn.Linear(feat_in, 2048, bias=True)
        self.fc_position = nn.Linear(feat_in, 3, bias=False)
        self.fc_rotation = nn.Linear(feat_in, 4, bias=False)

        init_modules = [self.fc_position, self.fc_rotation]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight.data)
                if module.bias is not None:
                    nn.init.constant(module.bias.data, 0)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation


class GoogleNet(nn.Module):
    """ PoseNet using Inception V3 """

    def __init__(self, base_model, fixed_weight=False, dropout_rate=0.0):
        super(GoogleNet, self).__init__()
        self.dropout_rate = dropout_rate

        model = []
        model.append(base_model.Conv2d_1a_3x3)
        model.append(base_model.Conv2d_2a_3x3)
        model.append(base_model.Conv2d_2b_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Conv2d_3b_1x1)
        model.append(base_model.Conv2d_4a_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Mixed_5b)
        model.append(base_model.Mixed_5c)
        model.append(base_model.Mixed_5d)
        model.append(base_model.Mixed_6a)
        model.append(base_model.Mixed_6b)
        model.append(base_model.Mixed_6c)
        model.append(base_model.Mixed_6d)
        model.append(base_model.Mixed_6e)
        model.append(base_model.Mixed_7a)
        model.append(base_model.Mixed_7b)
        model.append(base_model.Mixed_7c)
        self.base_model = nn.Sequential(*model)

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.pos = nn.Linear(2048, 3, bias=True)
        self.ori = nn.Linear(2048, 4, bias=True)

    def forward(self, x):
        # 299 x 299 x 3
        x = self.base_model(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        pos = self.pos(x)
        ori = self.ori(x)

        return pos, ori
