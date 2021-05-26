import os
import sys
from datetime import datetime

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torchsummary import summary
from torch.utils.data import DataLoader

sys.path.append("../")

from models.Loss import ClassLoss, BBoxLoss, LandmarkLoss, accuracy
from models.RNet import RNet
from utils.data import CustomDataset

# 设置损失值的比例
radio_cls_loss = 1.0
radio_bbox_loss = 0.5
radio_landmark_loss = 0.5

# 训练参数值
data_path = '../dataset/24/all_data'
batch_size = 384
learning_rate = 1e-3
epoch_num = 22
model_path = '../infer_models'

# 获取R模型
device = torch.device("cuda")
model = RNet()
model.to(device)
summary(model, (3, 24, 24))

# 获取数据
train_dataset = CustomDataset(data_path)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 设置优化方法
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-4)

# 获取学习率衰减函数
scheduler = MultiStepLR(optimizer, milestones=[6, 14, 20], gamma=0.1)

# 获取损失函数
class_loss = ClassLoss()
bbox_loss = BBoxLoss()
landmark_loss = LandmarkLoss()

# 开始训练
for epoch in range(epoch_num):
    for batch_id, (img, label, bbox, landmark) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device).long()
        bbox = bbox.to(device)
        landmark = landmark.to(device)
        class_out, bbox_out, landmark_out = model(img)
        cls_loss = class_loss(class_out, label)
        box_loss = bbox_loss(bbox_out, bbox, label)
        landmarks_loss = landmark_loss(landmark_out, landmark, label)
        total_loss = radio_cls_loss * cls_loss + radio_bbox_loss * box_loss + radio_landmark_loss * landmarks_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if batch_id % 100 == 0:
            acc = accuracy(class_out, label)
            print('[%s] Train epoch %d, batch %d, total_loss: %f, cls_loss: %f, box_loss: %f, landmarks_loss: %f, '
                  'accuracy：%f' % (
                  datetime.now(), epoch, batch_id, total_loss, cls_loss, box_loss, landmarks_loss, acc))
    scheduler.step()

    # 保存模型
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model, os.path.join(model_path, 'RNet.pth'))
