import torch.nn as nn


# 定义ResNet_Block
class ResNet_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet_Block, self).__init__()
        self.refPadding = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批量归一化
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)  # 批量归一化

    def forward(self, x):
        residual = x
        out = self.refPadding(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.refPadding(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = residual + out
        return out

# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 3层卷积层
        model = [
            nn.ReflectionPad2d(3),  # 反射填充，增加张量尺寸，保持图像边缘信息
            nn.Conv2d(3, 64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]

        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        ]

        model += [
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        ]

        # 6层ResNet
        for i in range(6):
            model += [
                ResNet_Block(256, 256)
            ]

        # 3层上采样
        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        ]

        model += [
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.conv1(x), 0.2)
        x = nn.functional.leaky_relu(self.conv2(x), 0.2)
        x = nn.functional.leaky_relu(self.conv3(x), 0.2)
        x = nn.functional.sigmoid(self.conv4(x))
        return x
