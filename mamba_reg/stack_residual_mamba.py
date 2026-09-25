import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import Decoder
from .encoder import Encoder
from .mamba import Mamba3D
from .spatial_transformer import SpatialTransformer2D


class StackResidualMamba(nn.Module):
    """
    Conservative stack-level residual refinement.

    The model works after an explicit local stack registration has already
    produced a stable volume. It predicts a small per-slice 2D residual field
    using true z-context rather than pair-wise F=1 context.
    """

    def __init__(
        self,
        feature_dim=24,
        mamba_depth=2,
        residual_max_flow=1.5,
        preserve_resolution=False,
        preserve_z_resolution=True,
        bidirectional_mamba=False,
    ):
        super().__init__()
        self.residual_max_flow = float(residual_max_flow)
        if preserve_resolution:
            down_stride = up_stride = (1, 1, 1)
        elif preserve_z_resolution:
            down_stride = up_stride = (1, 2, 2)
        else:
            down_stride = up_stride = (2, 2, 2)

        self.encoder = Encoder(in_channels=1, feature_dim=feature_dim, down_stride=down_stride)
        self.mamba = Mamba3D(
            d_model=feature_dim,
            depth=mamba_depth,
            bidirectional=bidirectional_mamba,
        )
        self.decoder = Decoder(in_channels=feature_dim, out_channels=2, up_stride=up_stride)
        self.stn = SpatialTransformer2D()

    def predict_flow(self, volume):
        _, _, frames, height, width = volume.shape
        feat = self.encoder(volume)
        batch, channels, feat_frames, feat_height, feat_width = feat.shape
        feat_seq = feat.permute(0, 2, 3, 4, 1).reshape(
            batch * feat_frames,
            feat_height * feat_width,
            channels,
        )
        feat_seq = self.mamba(
            feat_seq,
            video_length=feat_frames,
            height=feat_height,
            weight=feat_width,
        )
        feat = feat_seq.reshape(
            batch,
            feat_frames,
            feat_height,
            feat_width,
            channels,
        ).permute(0, 4, 1, 2, 3)
        flow = self.decoder(feat)
        flow = F.interpolate(
            flow,
            size=(frames, height, width),
            mode="trilinear",
            align_corners=True,
        )
        return torch.tanh(flow) * self.residual_max_flow

    def forward(self, volume):
        flow = self.predict_flow(volume)
        refined = self.stn(volume, flow)
        return refined, flow
