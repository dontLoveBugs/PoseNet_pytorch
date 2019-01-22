# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/21 19:48
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseLoss(nn.Module):
    def __init__(self, sx=0.0, sq=0.0, learn_beta=False):
        super(PoseLoss, self).__init__()
        self.learn_beta = learn_beta

        if not self.learn_beta:
            self.sx = 0
            self.sq = -6.25

        self.sx = nn.Parameter(torch.Tensor([sx]), requires_grad=self.learn_beta)
        self.sq = nn.Parameter(torch.Tensor([sq]), requires_grad=self.learn_beta)

        self.loss_print = None

    def forward(self, pred_x, pred_q, target_x, target_q):
        pred_q = F.normalize(pred_q, p=2, dim=1)
        loss_x = F.l1_loss(pred_x, target_x)
        loss_q = F.l1_loss(pred_q, target_q)

        loss = torch.exp(-self.sx) * loss_x + self.sx + torch.exp(-self.sq) * loss_q + self.sq

        self.loss_print = [loss.item(), loss_x.item(), loss_q.item()]

        return loss, loss_x.item(), loss_q.item()
