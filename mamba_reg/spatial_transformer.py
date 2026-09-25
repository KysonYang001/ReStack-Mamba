import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer2D(nn.Module):

    """
    moving:
        [B,1,F,H,W]

    flow:
        [B,2,F,H,W]

    output:
        [B,1,F,H,W]
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        moving,
        flow
    ):

        B, C, Fnum, H, W = moving.shape

        device = moving.device

        ##################################################
        # base grid
        ##################################################

        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )

        grid = torch.stack(
            [xx, yy],
            dim=0
        ).float()

        grid = grid.unsqueeze(0).unsqueeze(2)

        # [1,2,1,H,W]

        grid = grid.repeat(
            B,
            1,
            Fnum,
            1,
            1
        )

        ##################################################
        # add flow
        ##################################################

        new_grid = grid + flow

        ##################################################
        # normalize
        ##################################################

        x = new_grid[:, 0]

        y = new_grid[:, 1]

        x = 2.0 * x / (W - 1) - 1.0
        y = 2.0 * y / (H - 1) - 1.0

        sample_grid = torch.stack(
            [x, y],
            dim=-1
        )

        # [B,F,H,W,2]

        ##################################################
        # warp slice-by-slice
        ##################################################

        moving = moving.permute(
            0,
            2,
            1,
            3,
            4
        )

        moving = moving.reshape(
            B * Fnum,
            C,
            H,
            W
        )

        sample_grid = sample_grid.reshape(
            B * Fnum,
            H,
            W,
            2
        )

        warped = F.grid_sample(
            moving,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )

        warped = warped.reshape(
            B,
            Fnum,
            C,
            H,
            W
        )

        warped = warped.permute(
            0,
            2,
            1,
            3,
            4
        )

        return warped