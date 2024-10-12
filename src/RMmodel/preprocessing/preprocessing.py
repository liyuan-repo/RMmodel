import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.misc import lower_config
from src.config.default import get_cfg_defaults


# from torchsummary import summary


class DWConv1(nn.Module):
    """Depthwise conv + Pointwise conv"""

    def __init__(self, in_channels, out_channels):
        super(DWConv1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=(1, 1), stride=(1, 1), padding=0, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class DWConv3(nn.Module):
    """Depthwise conv + Pointwise conv"""

    def __init__(self, in_channels, out_channels):
        super(DWConv3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class DWConv5(nn.Module):
    """Depthwise conv + Pointwise conv"""

    def __init__(self, in_channels, out_channels):
        super(DWConv5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=(5, 5), stride=(1, 1), padding=2, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class DWConv7(nn.Module):
    """Depthwise conv + Pointwise conv"""

    def __init__(self, in_channels, out_channels):
        super(DWConv7, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=(7, 7), stride=(1, 1), padding=3, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


# feature preprocessing module (FPM)
class preprocess(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']

        self.DWConv1 = DWConv1(self.in_channels, self.out_channels)
        self.DWConv3 = DWConv3(self.in_channels, self.out_channels)
        self.DWConv5 = DWConv5(self.in_channels, self.out_channels)
        self.DWConv7 = DWConv7(self.in_channels, self.out_channels)
        # self.DWConv1 = DWConv1(256, 64)
        # self.DWConv3 = DWConv3(256, 64)
        # self.DWConv5 = DWConv5(256, 64)
        # self.DWConv7 = DWConv7(256, 64)

    def forward(self, x):
        FPM1 = self.DWConv1(x)
        FPM2 = self.DWConv3(x)
        FPM3 = self.DWConv5(x)
        FPM4 = self.DWConv7(x)
        FPM = torch.concat([FPM1, FPM2, FPM3, FPM4], dim=1)
        # self.FPM = FPM
        # print(self.FPM.size())
        # return torch.concat([FPM1, FPM2, FPM3, FPM4], dim=0)
        return FPM


if __name__ == "__main__":
    config = get_cfg_defaults()
    _config = lower_config(config)
    input = torch.randn(1, 256, 64, 64)
    device = torch.device("cuda")
    block = preprocess(_config['rmmodel']['preprocess'])
    result = block(input)
    print(result.size())

    # block1 = DWConv1(64, 64)
    # block3 = DWConv3(64, 64)
    # block5 = DWConv5(64, 64)
    # out1 = block1(input)
    # out3 = block3(input)
    # cc = torch.concat([out1, out3], dim=0)
    #
    # print(out1.size())
    # print(out3.size())
    # print(cc.size())

    # device = torch.device("cuda")
    # model1 = Block(64, 64, 3).to(device)
    # summary(model1, (64, 64, 64))
