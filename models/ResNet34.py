import torch
from torch import nn


# 因为ResNet34包含重复的单元，故用ResidualBlock类来简化代码
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),  # 要采样的话在这里改变stride
            nn.BatchNorm2d(outchannel),  # 批处理正则化
            nn.ReLU(inplace=True),  # 激活
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),  # 采样之后注意保持feature map的大小不变
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = shortcut

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.basic(x)
        residual = x if self.shortcut is None else self.shortcut(x)  # 计算残差
        out += residual
        return nn.ReLU(inplace=True)(out)  # 注意激活


# ResNet类
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )  # 开始的部分
        self.body = self.makelayers([3, 4, 6, 3])  # 具有重复模块的部分

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def makelayers(self, blocklist):  # 注意传入列表而不是解列表
        self.layers = []
        for index, blocknum in enumerate(blocklist):
            if index != 0:
                shortcut = nn.Sequential(
                    nn.Conv2d(64 * 2 ** (index - 1), 64 * 2 ** index, 1, 2, bias=False),
                    nn.BatchNorm2d(64 * 2 ** index)
                )  # 使得输入输出通道数调整为一致
                self.layers.append(ResidualBlock(64 * 2 ** (index - 1), 64 * 2 ** index, 2, shortcut))  # 每次变化通道数时进行下采样
            for i in range(0 if index == 0 else 1, blocknum):
                self.layers.append(ResidualBlock(64 * 2 ** index, 64 * 2 ** index, 1))
        return nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.body(x)
        x = nn.AvgPool2d(4)(x)
        x = x.view(x.size(0), -1)
        return x


def resnet34():
    return ResNet()


if __name__ == '__main__':
    net = ResNet()
    x = torch.zeros([2, 3, 112, 112])
    y = net(x)
    print(y.shape)
