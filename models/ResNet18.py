#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/30 23:19
# @Author  : Liu Lihao
# @File    : resnet.py

import torch
from torch import nn
from torch.nn import functional as F

'''
ResBlock
'''


class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        # 通过stride减少参数维度
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        # [b, ch_in, h, w] => [b, ch_out, h, w]
        self.extra = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
            nn.BatchNorm2d(ch_out)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''
        :param x: [b, ch, h, w]
        :return:
        '''
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out

        return out


'''
ResNet-18
'''


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        # follow 4 blocks
        # [b, 64, h, w] => [b, 128, h/2, w/2]
        self.blk1 = ResBlk(64, 128, stride=2)
        # [b, 128, h/2, w/2] => [b, 256, h/4, w/4]
        self.blk2 = ResBlk(128, 256, stride=2)
        # [b, 256, h/4, w/4] => [b, 512, h/8, w/8]
        self.blk3 = ResBlk(256, 512, stride=2)
        # [b, 512, h/8, w/8] => [b, 512, h/16, w/16]
        self.blk4 = ResBlk(512, 512, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # [b, 3, h, w] => [b, 64, h, w]
        x = F.relu(self.conv1(x))
        # [b, 64, h, w] => [b, 512, h/16, w/16]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # [b, 512, h/16, w/16] => [b, 512, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])

        # [b, 512, 1, 1] => [b, 512]
        x = x.view(x.size(0), -1)

        return x


def resnet18():
    return ResNet18()


if __name__ == '__main__':
    net = ResNet18()
    x = torch.zeros([2, 3, 112, 112])
    print(net(x).shape)
