import torch
import torch.nn as nn


class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        up_stride=(1, 2, 2),
    ):
        super().__init__()

        if up_stride == 1 or up_stride == (1, 1, 1):
            self.up = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        else:
            self.up = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=up_stride,
                stride=up_stride
            )

        self.conv = nn.Sequential(
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):

        x = self.up(x)

        x = self.conv(x)

        return x


class Decoder(nn.Module):

    """
    Input with the default anisotropic stride:
        [B, in_channels, F, H/4, W/4]

    Output:
        [B, out_channels, F, H, W]

    For vEM serial registration:
        out_channels = 2
        channel 0 = dx
        channel 1 = dy
    """

    def __init__(
        self,
        in_channels=64,
        out_channels=2,
        up_stride=(1, 2, 2),
    ):
        super().__init__()

        self.up1 = UpBlock(
            in_channels,
            64,
            up_stride=up_stride,
        )

        self.up2 = UpBlock(
            64,
            32,
            up_stride=up_stride,
        )

        self.flow_head = nn.Conv3d(
            32,
            out_channels,
            kernel_size=3,
            padding=1
        )

        nn.init.normal_(
            self.flow_head.weight,
            mean=0.0,
            std=1e-5
        )

        if self.flow_head.bias is not None:
            nn.init.constant_(
                self.flow_head.bias,
                0.0
            )

    def forward(self, x):

        x = self.up1(x)

        x = self.up2(x)

        flow = self.flow_head(x)

        return flow
