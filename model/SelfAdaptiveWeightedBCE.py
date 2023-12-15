# -*- coding: utf-8 -*-
# time: 2023/11/29 16:17
# file: SelfAdaptiveWeightedBCE.py
# author: Tommy Joe +
# Project: AERNet

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAdaptiveWeightedBCE(nn.Module):

    def __init__(self):
        super(SelfAdaptiveWeightedBCE, self).__init__()

    def forward(self, inputs, targets):
        # print("inputs:{}".format(inputs.shape))
        pred = torch.where(torch.sigmoid(inputs) > 0.5, 1, 0)
        w1, w2 = loss_weight(pred, targets)
        weight1 = torch.zeros_like(targets)
        weight1 = torch.fill_(weight1, w1)
        # print("targets:{}".format(targets))
        # print("w1:{},w2:{}".format(w1, w2))
        # print(weight1)
        weight1[targets > 0] = w2
        # print(targets > 0)
        # print(weight1)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=weight1, reduction="mean")
        return loss


def loss_weight(pred, targets):
    # TP是准确识别出变化像素的数量，FP是未变化像素被错误检测为变化像素的数量。 TN表示正确检测到的未变化像素的数量，FN表示被错误识别为未变化像素的变化像素的数量。(变化为1,未变为0)
    # 1.计算TP、FP、TN、FN
    # print("pred:{}".format(pred.shape))
    # print("targets:{}".format(targets.shape))
    # pred:torch.Size([24, 1, 16, 16])
    # targets:torch.Size([24, 1, 256, 256])
    TP = torch.sum(targets * pred)
    FP = torch.sum((1 - targets) * pred)
    TN = torch.sum((1 - targets) * (1 - pred))
    FN = torch.sum(targets * (1 - pred))  # 变化像素:targets=1,错误识别为未变化像素:pred=0
    # 计算w1 和 w2
    w1 = TN / (TN + FN + FP)
    w2 = TP / (TN + FP + FN)
    return w1, w2

def calculate_point(pred, targets):
    # TP是准确识别出变化像素的数量，FP是未变化像素被错误检测为变化像素的数量。 TN表示正确检测到的未变化像素的数量，FN表示被错误识别为未变化像素的变化像素的数量。(变化为1,未变为0)
    # 1.计算TP、FP、TN、FN
    # print("pred:{}".format(pred.shape))
    # print("targets:{}".format(targets.shape))
    # pred:torch.Size([24, 1, 16, 16])
    # targets:torch.Size([24, 1, 256, 256])
    TP = torch.sum(targets * pred)
    FP = torch.sum((1 - targets) * pred)
    TN = torch.sum((1 - targets) * (1 - pred))
    FN = torch.sum(targets * (1 - pred))  # 变化像素:targets=1,错误识别为未变化像素:pred=0

    return TP, FP, TN, FN


if __name__ == '__main__':
    inputs = torch.randn(4, 4)
    targets = torch.randn(4, 4)
    loss = SelfAdaptiveWeightedBCE()
    print(loss(inputs, targets))
