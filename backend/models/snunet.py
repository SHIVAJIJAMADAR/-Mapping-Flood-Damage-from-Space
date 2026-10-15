"""
SNUNet-CD architecture with Ensemble Channel Attention Module (ECAM)
Adapted from Kuro Siwo (NeurIPS 2024, Bountos et al.) & Open-CD (Fang et al.).
Optimized for multi-temporal SAR flood and change detection.

AUTHORITATIVE KURO SIWO GROUND TRUTH SPECIFICATION:
==================================================
Target Classes (from official Kuro Siwo dataset mask.npy):
  - Class 0: No water (land / background / no change)
  - Class 1: Permanent water (rivers, lakes, reservoirs)
  - Class 2: Flood water (flooded vegetation, inundated terrain, debris flows)

Input Contract:
  - xA (pre-event):  Tensor [B, 3, H, W] -> [Normalized Linear VV, Normalized Linear VH, Auxiliary (DEM/Ratio)]
  - xB (post-event): Tensor [B, 3, H, W] -> [Normalized Linear VV, Normalized Linear VH, Auxiliary (DEM/Ratio)]
  - Training spatial patch size: 224 x 224 (fully convolutional, supports any multiples of 16)
  - Output: Logits [B, 3, H, W]
"""

from typing import Dict
import torch
import torch.nn as nn

# Authoritative Kuro Siwo Class Constants
CLASS_NO_WATER: int = 0
CLASS_PERMANENT_WATER: int = 1
CLASS_FLOOD_WATER: int = 2

CLASS_NAMES: Dict[int, str] = {
    CLASS_NO_WATER: "No water (land/background)",
    CLASS_PERMANENT_WATER: "Permanent water",
    CLASS_FLOOD_WATER: "Flood water",
}

# Kuro Siwo Training Normalization Constants (Linear Backscatter Power)
IMG_MEANS = [0.0953, 0.0264]   # Linear VV mean, Linear VH mean
IMG_STDS = [0.0427, 0.0215]    # Linear VV std, Linear VH std

# Auxiliary Elevation & Slope Normalization
DEM_MEAN: float = 67.0293       # DEM elevation in meters
DEM_STD: float = 1765.0062      # DEM elevation std in meters
SLOPE_MEAN: float = 2.9482      # Terrain slope in degrees
SLOPE_STD: float = 79.2493      # Terrain slope std


class ConvBlockNested(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super(ConvBlockNested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.conv1(x)
        x = self.bn1(identity)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.activation(x + identity)


class UpSampleBlock(nn.Module):
    def __init__(self, in_ch: int, bilinear: bool = False):
        super(UpSampleBlock, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced_ch = max(1, in_channels // ratio)
        self.fc1 = nn.Conv2d(in_channels, reduced_ch, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_ch, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SNUNet_ECAM(nn.Module):
    """
    Siamese Densely Connected Network with Ensemble Channel Attention Module.
    Takes two multi-channel images (pre-event xA, post-event xB) and predicts class logits.
    """
    def __init__(self, in_channels: int = 3, out_ch: int = 3, base_channel: int = 32):
        super(SNUNet_ECAM, self).__init__()
        n1 = base_channel
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoders
        self.conv0_0 = ConvBlockNested(in_channels, filters[0], filters[0])
        self.conv1_0 = ConvBlockNested(filters[0], filters[1], filters[1])
        self.Up1_0 = UpSampleBlock(filters[1])
        self.conv2_0 = ConvBlockNested(filters[1], filters[2], filters[2])
        self.Up2_0 = UpSampleBlock(filters[2])
        self.conv3_0 = ConvBlockNested(filters[2], filters[3], filters[3])
        self.Up3_0 = UpSampleBlock(filters[3])
        self.conv4_0 = ConvBlockNested(filters[3], filters[4], filters[4])
        self.Up4_0 = UpSampleBlock(filters[4])

        # Dense Nested Blocks (Decoder 1)
        self.conv0_1 = ConvBlockNested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_1 = ConvBlockNested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.Up1_1 = UpSampleBlock(filters[1])
        self.conv2_1 = ConvBlockNested(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.Up2_1 = UpSampleBlock(filters[2])
        self.conv3_1 = ConvBlockNested(filters[3] * 2 + filters[4], filters[3], filters[3])
        self.Up3_1 = UpSampleBlock(filters[3])

        # Dense Nested Blocks (Decoder 2)
        self.conv0_2 = ConvBlockNested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_2 = ConvBlockNested(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.Up1_2 = UpSampleBlock(filters[1])
        self.conv2_2 = ConvBlockNested(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.Up2_2 = UpSampleBlock(filters[2])

        # Dense Nested Blocks (Decoder 3 & 4)
        self.conv0_3 = ConvBlockNested(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_3 = ConvBlockNested(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.Up1_3 = UpSampleBlock(filters[1])

        self.conv0_4 = ConvBlockNested(filters[0] * 5 + filters[1], filters[0], filters[0])

        # Ensemble Channel Attention
        self.ca = ChannelAttention(filters[0] * 4, ratio=16)
        self.ca1 = ChannelAttention(filters[0], ratio=16 // 4)

        # Final Classifier
        self.conv_final = nn.Conv2d(filters[0] * 4, out_ch, kernel_size=1)

    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        """
        xA: Pre-event tensor of shape (B, C, H, W)
        xB: Post-event tensor of shape (B, C, H, W)
        returns: Logits of shape (B, out_ch, H, W)
        """
        # Encoder branch A (Pre-event)
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))

        # Encoder branch B (Post-event)
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        # Nested connections level 1
        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))

        # Nested connections level 2
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))

        # Nested connections level 3
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))

        # Nested connections level 4
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))

        # Fusion & Attention
        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)
        intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))

        return self.conv_final(out)
