import torch
import numpy as np
import torch.nn as nn
from torch.nn import init


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):
    def __init__(self, version=1.0):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),  # (96, 96, 109, 109)
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # (96, 96, 54, 54)
                Fire(96, 16, 64, 64),  # (96, 128, 54, 54)
                Fire(128, 16, 64, 64),  # (96, 128, 54, 54)
                Fire(128, 32, 128, 128),  # (96, 256, 54, 54)
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # (96, 256, 27, 27)
                Fire(256, 32, 128, 128),  # (96, 256, 27, 27)
                Fire(256, 48, 192, 192),  # (96, 384, 27, 27)
                Fire(384, 48, 192, 192),  # (96, 384, 27, 27)
                Fire(384, 64, 256, 256),  # (96, 512, 27, 27)
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # (96, 512, 13, 13)
                Fire(512, 64, 256, 256),  # (96, 512, 13, 13)
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


if __name__ == '__main__':
    x = np.random.rand(1, 3, 112, 112)
    x = torch.tensor(x, dtype=torch.float32)
    model = SqueezeNet()
    x = model(x)
    print(x.shape)

    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    model = SqueezeNet()
    tensor = (torch.rand(1, 3, 112, 112),)
    flops = FlopCountAnalysis(model, tensor)
    print("FLOPs: {:.2e}".format(flops.total()))
    print(parameter_count_table(model))
