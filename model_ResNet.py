import torch.nn as nn
import yaml


option_path = 'configs/config.yml'
with open(option_path, 'r') as file_option:
    option = yaml.safe_load(file_option)


class ResBlock(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.conv0 = nn.Conv2d(nc, nc, kernel_size=3, padding=1)
        self.norm0 = nn.BatchNorm2d(nc)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(nc, nc, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(nc)

    def forward(self, x):
        out = self.conv0(x)
        out = self.norm0(out)
        out = self.act(out)

        out = self.conv1(out)
        out = self.norm1(out)

        return self.act(x + out)


class ResTruck(nn.Module):
    def __init__(self, nc, num_blocks):
        super().__init__()
        truck = []

        for i in range(num_blocks):
            truck += [ResBlock(nc)]
        self.truck = nn.Sequential(*truck)

    def forward(self, x):
        return self.truck(x)


class ResNet(nn.Module):
    def __init__(self, in_nc, nc, out_nc):
        super().__init__()
        self.conv0 = nn.Conv2d(in_nc, nc, kernel_size=7, stride=2)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.maxpool = nn.MaxPool2d(2,2)

        self.layer1 = ResTruck(nc, 3)
        self.conv1 = nn.Conv2d(nc, 2*nc, 3, padding=1, stride=2)

        self.layer2 = ResTruck(2*nc, 4)
        self.conv2 = nn.Conv2d(2*nc, 4*nc, 3, padding=1, stride=2)

        self.layer3 = ResTruck(4*nc, 6)
        self.conv3 = nn.Conv2d(4*nc, 4*nc, 3, padding=1, stride=2)

        self.layer4 = ResTruck(4*nc, 3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(4*nc, out_nc)

    def forward(self, x):
        out = self.conv0(x)
        out = self.act(out)
        out = self.maxpool(out)
        out = self.layer1(out)

        out = self.conv1(out)
        out = self.layer2(out)
        out = self.conv2(out)
        out = self.layer3(out)
        out = self.conv3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear(out)

        return out