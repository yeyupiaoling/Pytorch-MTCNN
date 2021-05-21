import torch.nn as nn
import torch
import numpy as np


class ClassLoss(nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()
        self.entropy_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.keep_ratio = 0.7

    def forward(self, class_out, label):
        # 保留neg 0 和pos 1 的数据，忽略掉part -1, landmark -2
        zeros = torch.zeros_like(label)
        ignore_label = torch.full_like(label, fill_value=-100)
        label = torch.where(torch.less(label, zeros), ignore_label, label)
        # 求neg 0 和pos 1 的数据70%数据
        ones = torch.ones_like(label)
        valid_label = torch.where(torch.greater_equal(label, zeros), ones, zeros)
        num_valid = torch.sum(valid_label)
        keep_num = int((num_valid * self.keep_ratio).numpy()[0])
        # 计算交叉熵损失
        loss = self.entropy_loss(input=class_out, label=label)
        # 取有效数据的70%计算损失
        loss, _ = torch.topk(torch.squeeze(loss), k=keep_num)
        return torch.mean(loss)


class BBoxLoss(nn.Module):
    def __init__(self):
        super(BBoxLoss, self).__init__()
        self.square_loss = nn.MSELoss(reduction='none')
        self.keep_ratio = 1.0

    def forward(self, bbox_out, bbox_target, label):
        # 保留pos 1 和part -1 的数据
        ones = torch.ones_like(label)
        zeros = torch.zeros_like(label)
        valid_label = torch.where(torch.equal(torch.abs(label), ones), ones, zeros)
        valid_label = torch.squeeze(valid_label)
        # 获取有效值的总数
        keep_num = int(torch.sum(valid_label).numpy()[0] * self.keep_ratio)
        loss = self.square_loss(input=bbox_out, label=bbox_target)
        loss = torch.sum(loss, dim=1)
        loss = loss * valid_label
        # 取有效数据计算损失
        loss, _ = torch.topk(loss, k=keep_num, dim=0)
        return torch.mean(loss)


class LandmarkLoss(nn.Module):
    def __init__(self):
        super(LandmarkLoss, self).__init__()
        self.square_loss = nn.MSELoss(reduction='none')
        self.keep_ratio = 1.0

    def forward(self, landmark_out, landmark_target, label):
        # 只保留landmark数据 -2
        ones = torch.ones_like(label)
        zeros = torch.zeros_like(label)
        valid_label = torch.where(torch.equal(label, torch.full_like(label, fill_value=-2)), ones, zeros)
        valid_label = torch.squeeze(valid_label)
        # 获取有效值的总数
        keep_num = int(torch.sum(valid_label).numpy()[0] * self.keep_ratio)
        loss = self.square_loss(input=landmark_out, label=landmark_target)
        loss = torch.sum(loss, dim=1)
        loss = loss * valid_label
        # 取有效数据计算损失
        loss, _ = torch.topk(loss, k=keep_num, dim=0)
        return torch.mean(loss)


# 求训练时的准确率
def accuracy(class_out, label):
    # 查找neg 0 和pos 1所在的位置
    zeros = torch.zeros_like(label)
    cond = torch.greater_equal(label, zeros)
    picked, _ = np.where(cond.numpy())
    picked = torch.to_tensor(picked, dtype='int32')
    # 求neg 0 和pos 1的准确率
    valid_class_out = torch.gather(class_out, picked)
    valid_label = torch.gather(label, picked)
    acc = torch.metric.accuracy(valid_class_out, valid_label)
    return acc
