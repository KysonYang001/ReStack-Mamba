import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):

    """
    Input
        [B,1,F,H,W]

    Output with the default anisotropic stride
        [B,feature_dim,F,H/4,W/4]
    """

    def __init__(
        self,
        in_channels=1,
        feature_dim=64,
        down_stride=(1, 2, 2),
    ):
        super().__init__()

        self.conv1 = ConvBlock(
            in_channels,
            32
        )

        self.down1 = nn.Conv3d(
            32,
            64,
            kernel_size=3,
            stride=down_stride,
            padding=1
        )

        self.conv2 = ConvBlock(
            64,
            64
        )

        self.down2 = nn.Conv3d(
            64,
            feature_dim,
            kernel_size=3,
            stride=down_stride,
            padding=1
        )

        self.conv3 = ConvBlock(
            feature_dim,
            feature_dim
        )

    def forward(self, x):

        x = self.conv1(x)

        x = self.down1(x)

        x = self.conv2(x)

        x = self.down2(x)

        x = self.conv3(x)

        return x
