import torch.nn as nn
import torch


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3))
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3))
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.prelu3 = nn.PReLU()
        self.conv4_1 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1, 1))
        self.conv4_2 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=(1, 1))
        self.conv4_3 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1))

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        # 分类是否人脸的卷积输出层
        class_out = self.conv4_1(x)
        class_out = torch.squeeze(class_out, dim=2)
        class_out = torch.squeeze(class_out, dim=2)
        # 人脸box的回归卷积输出层
        bbox_out = self.conv4_2(x)
        bbox_out = torch.squeeze(bbox_out, dim=2)
        bbox_out = torch.squeeze(bbox_out, dim=2)
        # 5个关键点的回归卷积输出层
        landmark_out = self.conv4_3(x)
        landmark_out = torch.squeeze(landmark_out, dim=2)
        landmark_out = torch.squeeze(landmark_out, dim=2)
        return class_out, bbox_out, landmark_out
