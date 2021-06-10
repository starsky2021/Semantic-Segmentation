import torch
import torch.nn.functional as F
from torch import nn

from utils import initialize_weights


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dilation=0):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        out = self.pool1(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class MyNet(nn.Module):
    def __init__(self, num_classes):
        super(MyNet, self).__init__()
        self.inconv1 = nn. Conv2d(3, 64, kernel_size=3, stride=1, padding=1, dilation=1)  # 244
        self.relu = nn.ReLU(inplace=True)
        self.inbn1 = nn.BatchNorm2d(64)
        self.inconv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1)  # 244
        self.inbn2 = nn.BatchNorm2d(128)
        self.inconv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1)  # 244
        self.inbn3 = nn.BatchNorm2d(256)
        self.inconv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, dilation=1)  # 244
        self.inbn4 = nn.BatchNorm2d(512)

        self.enc1 = EncoderBlock(512, 512, 1, 1)  # 122
        self.enc2 = EncoderBlock(512, 512, 1, 1)  # 56
        self.enc3 = EncoderBlock(512, 512, 1, 1)  # 28
        self.enc4 = EncoderBlock(512, 512, 1, 1)  # 14
        self.center = DecoderBlock(512, 1024, 512)  # 28
        self.dec4 = DecoderBlock(1024, 512, 256)  # 56
        self.dec3 = DecoderBlock(768, 256, 128)    # 122
        self.dec2 = DecoderBlock(640, 128, 64)    # 244
        self.dec1 = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        inconv1 = self.inconv1(x)
        inbn1 = self.inbn1(inconv1)
        inrelu1 = self.relu(inbn1)

        inconv2 = self.inconv2(inrelu1)
        inbn2 = self.inbn2(inconv2)
        inrelu2 = self.relu(inbn2)

        inconv3 = self.inconv3(inrelu2)
        inbn3 = self.inbn3(inconv3)
        inrelu3 = self.relu(inbn3)

        inconv4 = self.inconv4(inrelu3)
        inbn4 = self.inbn4(inconv4)
        inrelu4 = self.relu(inbn4)

        enc1 = self.enc1(inrelu4)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)
        return F.upsample(final, x.size()[2:], mode='bilinear')

