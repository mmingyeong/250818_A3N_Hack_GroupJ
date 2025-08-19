
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
    
    Supports:
      - mode="single": Single Map, Single Simulator (default)
      - mode="multi": Multi-Map input with Shared Encoder + Fusion
    
    Input:
      mode="single" -> (N, C, 256, 256)
      mode="multi"  -> list of maps [ (N, C, 256, 256), (N, C, 256, 256), ... ]
    Output:
      (N, out_dim)
    """
    def __init__(self, in_channels=15, out_dim=6, mode="single", fusion="concat"):
        super().__init__()
        self.mode = mode
        self.fusion = fusion  # "concat", "mean", "sum" 등 선택 가능

        # Shared encoder
        self.enc1 = ConvBlockEnc2D(in_channels, 32)
        self.enc2 = ConvBlockEnc2D(32, 64)
        self.enc3 = ConvBlockEnc2D(64, 128)
        self.enc4 = ConvBlockEnc2D(128, 256)
        self.enc5 = ConvBlockEnc2D(256, 512)

        # Regression head
        # fusion 방법에 따라 input 크기 달라짐
        if self.fusion == "concat":
            self.fc_in_dim = 512 * (1 if self.mode == "single" else 2)  # 기본 2개 맵 가정
        else:
            self.fc_in_dim = 512

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.fc_in_dim, out_dim)

    def encode(self, x):
        """Shared encoder for single map."""
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.global_pool(x)  # (N, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (N, 512)
        return x

    def forward(self, x):
        if self.mode == "single":
            # Standard single-map input
            feat = self.encode(x)

        elif self.mode == "multi":
            # Expect list of maps: [map1, map2, ...]
            feats = [self.encode(m) for m in x]  # list of (N, 512)
            if self.fusion == "concat":
                feat = torch.cat(feats, dim=1)  # (N, 512 * num_maps)
            elif self.fusion == "mean":
                feat = torch.stack(feats, dim=0).mean(dim=0)  # (N, 512)
            elif self.fusion == "sum":
                feat = torch.stack(feats, dim=0).sum(dim=0)
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion}")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Regression head
        out = self.fc(feat)  # (N, out_dim)
        return out
