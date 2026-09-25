import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalNCC2D(nn.Module):

    def __init__(self, win=9):
        super().__init__()
        self.win = win

    def forward(self, I, J):
        """
        I, J:
            [N,1,H,W]
        """

        win = self.win

        filt = torch.ones(
            1,
            1,
            win,
            win,
            device=I.device,
            dtype=I.dtype
        )

        padding = win // 2

        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = F.conv2d(I, filt, padding=padding)
        J_sum = F.conv2d(J, filt, padding=padding)
        I2_sum = F.conv2d(I2, filt, padding=padding)
        J2_sum = F.conv2d(J2, filt, padding=padding)
        IJ_sum = F.conv2d(IJ, filt, padding=padding)

        win_size = win * win

        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = (
            IJ_sum
            - u_J * I_sum
            - u_I * J_sum
            + u_I * u_J * win_size
        )

        I_var = (
            I2_sum
            - 2 * u_I * I_sum
            + u_I * u_I * win_size
        )

        J_var = (
            J2_sum
            - 2 * u_J * J_sum
            + u_J * u_J * win_size
        )

        cc = cross * cross / (
            I_var * J_var + 1e-5
        )

        return -torch.mean(cc)


class ChainSimilarityLoss(nn.Module):

    def __init__(
        self,
        use_ncc=True,
        win=9
    ):
        super().__init__()

        self.use_ncc = use_ncc

        if use_ncc:
            self.sim = LocalNCC2D(win=win)
        else:
            self.sim = nn.MSELoss()

    def forward(self, registered_volume):
        """
        registered_volume:
            [B,1,F,H,W]
        """

        B, C, Fnum, H, W = registered_volume.shape

        prev_slices = registered_volume[:, :, :-1]
        curr_slices = registered_volume[:, :, 1:]

        prev_slices = (
            prev_slices
            .permute(0, 2, 1, 3, 4)
            .reshape(B * (Fnum - 1), C, H, W)
        )

        curr_slices = (
            curr_slices
            .permute(0, 2, 1, 3, 4)
            .reshape(B * (Fnum - 1), C, H, W)
        )

        return self.sim(
            curr_slices,
            prev_slices
        )


class GradientLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, flow):
        """
        flow:
            [B,2,F-1,H,W]
        """

        dx = torch.abs(
            flow[:, :, :, :, 1:]
            -
            flow[:, :, :, :, :-1]
        )

        dy = torch.abs(
            flow[:, :, :, 1:, :]
            -
            flow[:, :, :, :-1, :]
        )

        dz = torch.abs(
            flow[:, :, 1:, :, :]
            -
            flow[:, :, :-1, :, :]
        )

        dx = dx * dx
        dy = dy * dy
        dz = dz * dz

        return (
            dx.mean()
            +
            dy.mean()
            +
            dz.mean()
        ) / 3.0


class MultiScaleNCC2D(nn.Module):
    def __init__(self, win=9, scales=(1.0, 0.5, 0.25)):
        super().__init__()
        self.scales = scales
        self.ncc = LocalNCC2D(win=win)

    def forward(self, moving, fixed):
        losses = []
        for scale in self.scales:
            if scale == 1.0:
                moving_s = moving
                fixed_s = fixed
            else:
                moving_s = F.interpolate(
                    moving,
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=True,
                    recompute_scale_factor=False,
                )
                fixed_s = F.interpolate(
                    fixed,
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=True,
                    recompute_scale_factor=False,
                )
            losses.append(self.ncc(moving_s, fixed_s))
        return torch.stack(losses).mean()


class SobelGradient(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0
        ky = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x):
        gx = F.conv2d(x, self.kx.to(dtype=x.dtype), padding=1)
        gy = F.conv2d(x, self.ky.to(dtype=x.dtype), padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)


class BendingEnergyLoss(nn.Module):
    def forward(self, flow):
        dxx = flow[:, :, :, :, 2:] - 2 * flow[:, :, :, :, 1:-1] + flow[:, :, :, :, :-2]
        dyy = flow[:, :, :, 2:, :] - 2 * flow[:, :, :, 1:-1, :] + flow[:, :, :, :-2, :]
        terms = []
        if dxx.numel():
            terms.append((dxx * dxx).mean())
        if dyy.numel():
            terms.append((dyy * dyy).mean())
        if flow.shape[-1] > 1 and flow.shape[-2] > 1:
            dxy = (
                flow[:, :, :, 1:, 1:]
                - flow[:, :, :, 1:, :-1]
                - flow[:, :, :, :-1, 1:]
                + flow[:, :, :, :-1, :-1]
            )
            terms.append(2.0 * (dxy * dxy).mean())
        if not terms:
            return flow.new_tensor(0.0)
        return torch.stack(terms).mean()


class EnhancedRegistrationLoss(nn.Module):
    def __init__(
        self,
        sim_weight=1.0,
        edge_weight=0.25,
        smooth_weight=0.02,
        bending_weight=0.005,
        magnitude_weight=0.0005,
        win=9,
    ):
        super().__init__()
        self.sim_weight = sim_weight
        self.edge_weight = edge_weight
        self.smooth_weight = smooth_weight
        self.bending_weight = bending_weight
        self.magnitude_weight = magnitude_weight

        self.ms_ncc = MultiScaleNCC2D(win=win)
        self.edge = SobelGradient()
        self.edge_ncc = MultiScaleNCC2D(win=win, scales=(1.0, 0.5))
        self.smooth = GradientLoss()
        self.bending = BendingEnergyLoss()

    def flatten_pairs(self, volume):
        batch, channels, frames, height, width = volume.shape
        return (
            volume.permute(0, 2, 1, 3, 4)
            .reshape(batch * frames, channels, height, width)
        )

    def forward(self, registered_volume, flow, source_volume=None):
        if source_volume is None:
            raise ValueError("EnhancedRegistrationLoss requires source_volume.")

        moving = registered_volume[:, :, 1:]
        fixed = source_volume[:, :, :-1]
        moving_2d = self.flatten_pairs(moving)
        fixed_2d = self.flatten_pairs(fixed)

        sim_loss = self.ms_ncc(moving_2d, fixed_2d)
        edge_loss = self.edge_ncc(self.edge(moving_2d), self.edge(fixed_2d))
        smooth_loss = self.smooth(flow)
        bending_loss = self.bending(flow)
        magnitude_loss = (flow * flow).mean()

        total = (
            self.sim_weight * sim_loss
            + self.edge_weight * edge_loss
            + self.smooth_weight * smooth_loss
            + self.bending_weight * bending_loss
            + self.magnitude_weight * magnitude_loss
        )

        return total, {
            "total": total,
            "sim": sim_loss,
            "edge": edge_loss,
            "smooth": smooth_loss,
            "bending": bending_loss,
            "magnitude": magnitude_loss,
        }


class RegistrationLoss(nn.Module):

    def __init__(
        self,
        sim_weight=1.0,
        smooth_weight=0.01,
        use_ncc=True,
        win=9
    ):
        super().__init__()

        self.sim_weight = sim_weight
        self.smooth_weight = smooth_weight

        self.chain_sim = ChainSimilarityLoss(
            use_ncc=use_ncc,
            win=win
        )

        self.smooth = GradientLoss()

    def forward(
        self,
        registered_volume,
        flow,
        source_volume=None
    ):

        sim_loss = self.chain_sim(
            registered_volume
        )

        smooth_loss = self.smooth(
            flow
        )

        total = (
            self.sim_weight * sim_loss
            +
            self.smooth_weight * smooth_loss
        )

        loss_dict = {
            "total": total,
            "sim": sim_loss,
            "smooth": smooth_loss
        }

        return total, loss_dict
