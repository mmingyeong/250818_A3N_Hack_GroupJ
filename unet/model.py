import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlockEnc2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
        super().__init__()
        self.pad = nn.ReplicationPad2d(2)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class UNetEncoderRegressor(nn.Module):
    """
    Encoder-only U-Net adapted for regression of cosmological parameters.
    Input: (N, in_channels, 256, 256)
    Output: (N, out_dim)
    """
    def __init__(self, in_channels=15, out_dim=6):
        super().__init__()
        # Encoding path
        self.enc1 = ConvBlockEnc2D(in_channels, 32)
        self.enc2 = ConvBlockEnc2D(32, 64)
        self.enc3 = ConvBlockEnc2D(64, 128)
        self.enc4 = ConvBlockEnc2D(128, 256)
        self.enc5 = ConvBlockEnc2D(256, 512)

        # Global pooling + regression head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        # Encoding path
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)

        # Global feature vector
        x = self.global_pool(x)  # (N, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (N, 512)

        # Fully connected to predict parameters
        out = self.fc(x)  # (N, out_dim)
        return out
