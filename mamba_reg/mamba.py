# mamba.py

import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    def __init__(
        self,
        dim,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))


class BidirectionalMambaBlock(MambaBlock):
    """Shared-weight forward/reverse scan with a residual-preserving merge."""

    def forward(self, x):
        normalized = self.norm(x)
        forward = self.mamba(normalized)
        reverse = torch.flip(
            self.mamba(torch.flip(normalized, dims=(1,))),
            dims=(1,),
        )
        return x + self.dropout(0.5 * (forward + reverse))


class WeightedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.weight_net = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, 3)
        )

    def forward(self, xy, xz, yz):
        """
        xy, xz, yz:
            [B*D, H*W, C]
        """

        feat = torch.cat(
            [xy, xz, yz],
            dim=-1
        )

        weight = self.weight_net(feat)
        weight = torch.softmax(weight, dim=-1)

        out = (
            xy * weight[..., 0:1]
            + xz * weight[..., 1:2]
            + yz * weight[..., 2:3]
        )

        return out


class Mamba3D(nn.Module):
    """
    vEM Registration Mamba3D

    Input:
        x:
            [B*D, H*W, C]

    Args:
        video_length:
            D
        height:
            H
        weight:
            W

    Output:
        out:
            [B*D, H*W, C]
    """

    def __init__(
        self,
        d_model,
        depth=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0,
        bidirectional=False,
    ):
        super().__init__()

        self.d_model = d_model

        block_class = BidirectionalMambaBlock if bidirectional else MambaBlock

        self.xy_blocks = nn.ModuleList([
            block_class(
                dim=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.xz_blocks = nn.ModuleList([
            block_class(
                dim=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.yz_blocks = nn.ModuleList([
            block_class(
                dim=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.fusion = WeightedFusion(d_model)

        self.out_norm = nn.LayerNorm(d_model)

    def to_volume(self, x, D, H, W):
        """
        [B*D, H*W, C] -> [B,D,H,W,C]
        """

        B = x.shape[0] // D

        x = rearrange(
            x,
            "(b d) (h w) c -> b d h w c",
            b=B,
            d=D,
            h=H,
            w=W
        )

        return x

    def to_tokens(self, x):
        """
        [B,D,H,W,C] -> [B*D, H*W, C]
        """

        x = rearrange(
            x,
            "b d h w c -> (b d) (h w) c"
        )

        return x

    def scan_xy(self, x):
        """
        XY scan:
            each slice independently

        [B,D,H,W,C] -> [B*D,H*W,C]
        """

        return rearrange(
            x,
            "b d h w c -> (b d) (h w) c"
        )

    def restore_xy(self, x, B, D, H, W):
        """
        [B*D,H*W,C] -> [B,D,H,W,C]
        """

        return rearrange(
            x,
            "(b d) (h w) c -> b d h w c",
            b=B,
            d=D,
            h=H,
            w=W
        )

    def scan_xz(self, x):
        """
        XZ scan:
            for each y row, scan over D x W

        [B,D,H,W,C] -> [B*H,D*W,C]
        """

        return rearrange(
            x,
            "b d h w c -> (b h) (d w) c"
        )

    def restore_xz(self, x, B, D, H, W):
        """
        [B*H,D*W,C] -> [B,D,H,W,C]
        """

        return rearrange(
            x,
            "(b h) (d w) c -> b d h w c",
            b=B,
            h=H,
            d=D,
            w=W
        )

    def scan_yz(self, x):
        """
        YZ scan:
            for each x column, scan over D x H

        [B,D,H,W,C] -> [B*W,D*H,C]
        """

        return rearrange(
            x,
            "b d h w c -> (b w) (d h) c"
        )

    def restore_yz(self, x, B, D, H, W):
        """
        [B*W,D*H,C] -> [B,D,H,W,C]
        """

        return rearrange(
            x,
            "(b w) (d h) c -> b d h w c",
            b=B,
            w=W,
            d=D,
            h=H
        )

    def forward(
        self,
        x,
        video_length,
        height,
        weight
    ):
        """
        x:
            [B*D,H*W,C]
        """

        D = video_length
        H = height
        W = weight

        B = x.shape[0] // D
        C = x.shape[-1]

        residual = x

        volume = self.to_volume(
            x,
            D,
            H,
            W
        )

        xy = self.scan_xy(volume)
        xz = self.scan_xz(volume)
        yz = self.scan_yz(volume)

        for blk in self.xy_blocks:
            xy = blk(xy)

        for blk in self.xz_blocks:
            xz = blk(xz)

        for blk in self.yz_blocks:
            yz = blk(yz)

        xy = self.restore_xy(
            xy,
            B,
            D,
            H,
            W
        )

        xz = self.restore_xz(
            xz,
            B,
            D,
            H,
            W
        )

        yz = self.restore_yz(
            yz,
            B,
            D,
            H,
            W
        )

        xy = self.to_tokens(xy)
        xz = self.to_tokens(xz)
        yz = self.to_tokens(yz)

        out = self.fusion(
            xy,
            xz,
            yz
        )

        out = self.out_norm(out + residual)

        return out


if __name__ == "__main__":
    model = Mamba3D(
        d_model=64,
        depth=2
    ).cuda()

    B = 2
    D = 8
    H = 16
    W = 16
    C = 64

    x = torch.randn(
        B * D,
        H * W,
        C
    ).cuda()

    y = model(
        x,
        video_length=D,
        height=H,
        weight=W
    )

    print("input:", x.shape)
    print("output:", y.shape)
