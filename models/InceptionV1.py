import torch
import torch.nn as nn


# Inception模块
class Block(nn.Module):
    def __init__(self, in_channels, out_chanel_1, out_channel_3_reduce, out_channel_3,
                 out_channel_5_reduce, out_channel_5, out_channel_pool):
        super(Block, self).__init__()

        block = []
        self.block1 = nn.Conv2d(in_channels=in_channels, out_channels=out_chanel_1, kernel_size=1)
        block.append(self.block1)
        self.block2_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channel_3_reduce, kernel_size=1)
        self.block2 = nn.Conv2d(in_channels=out_channel_3_reduce, out_channels=out_channel_3, kernel_size=3, padding=1)
        block.append(self.block2)
        self.block3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channel_5_reduce, kernel_size=1)
        self.block3 = nn.Conv2d(in_channels=out_channel_5_reduce, out_channels=out_channel_5, kernel_size=3, padding=1)
        block.append(self.block3)
        self.block4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.block4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channel_pool, kernel_size=1)
        block.append(self.block4)

        # self.incep = nn.Sequential(*block)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(self.block2_1(x))
        out3 = self.block3(self.block3_1(x))
        out4 = self.block4(self.block4_1(x))
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


# 在完整网络中间某层输出结果以一定的比例添加到最终结果分类
class InceptionClassifiction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionClassifiction, self).__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.linear1 = nn.Linear(in_features=128 * 4 * 4, out_features=1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.7)
        self.linear2 = nn.Linear(in_features=1024, out_features=out_channels)

    def forward(self, x):
        x = self.conv1(self.avgpool(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.linear1(x))
        out = self.linear2(self.dropout(x))
        return out


class InceptionV1(nn.Module):
    def __init__(self):
        super(InceptionV1, self).__init__()

        self.blockA = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(64),

        )
        self.blockB = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.LocalResponseNorm(192),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.blockC = nn.Sequential(
            Block(in_channels=192, out_chanel_1=64, out_channel_3_reduce=96, out_channel_3=128,
                  out_channel_5_reduce=16, out_channel_5=32, out_channel_pool=32),
            Block(in_channels=256, out_chanel_1=128, out_channel_3_reduce=128, out_channel_3=192,
                  out_channel_5_reduce=32, out_channel_5=96, out_channel_pool=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.blockD_1 = Block(in_channels=480, out_chanel_1=192, out_channel_3_reduce=96, out_channel_3=208,
                              out_channel_5_reduce=16, out_channel_5=48, out_channel_pool=64)

        self.blockD_2 = nn.Sequential(
            Block(in_channels=512, out_chanel_1=160, out_channel_3_reduce=112, out_channel_3=224,
                  out_channel_5_reduce=24, out_channel_5=64, out_channel_pool=64),
            Block(in_channels=512, out_chanel_1=128, out_channel_3_reduce=128, out_channel_3=256,
                  out_channel_5_reduce=24, out_channel_5=64, out_channel_pool=64),
            Block(in_channels=512, out_chanel_1=112, out_channel_3_reduce=144, out_channel_3=288,
                  out_channel_5_reduce=32, out_channel_5=64, out_channel_pool=64),
        )

        self.blockD_3 = nn.Sequential(
            Block(in_channels=528, out_chanel_1=256, out_channel_3_reduce=160, out_channel_3=320,
                  out_channel_5_reduce=32, out_channel_5=128, out_channel_pool=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.blockE = nn.Sequential(
            Block(in_channels=832, out_chanel_1=256, out_channel_3_reduce=160, out_channel_3=320,
                  out_channel_5_reduce=32, out_channel_5=128, out_channel_pool=128),
            Block(in_channels=832, out_chanel_1=384, out_channel_3_reduce=192, out_channel_3=384,
                  out_channel_5_reduce=48, out_channel_5=128, out_channel_pool=128),
        )

        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)

    def forward(self, x):
        x = self.blockA(x)
        x = self.blockB(x)
        x = self.blockC(x)
        x = self.blockD_1(x)
        x = self.blockD_2(x)
        x = self.blockD_3(x)
        out = self.blockE(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out


if __name__ == '__main__':
    model = InceptionV1()
    input = torch.randn(2, 3, 112, 112)
    y = model(input)
    print(y.shape)

    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    model = InceptionV1()
    tensor = (torch.rand(1, 3, 112, 112),)
    flops = FlopCountAnalysis(model, tensor)
    print("FLOPs: {:.2e}".format(flops.total()))
    print(parameter_count_table(model))
