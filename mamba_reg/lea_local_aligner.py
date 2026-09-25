import inspect
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


def coords_grid(batch, height, width, device):
    coords = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def mesh_grid(batch, height, width):
    x_base = torch.arange(0, width).repeat(batch, height, 1)
    y_base = torch.arange(0, height).repeat(batch, width, 1).transpose(1, 2)
    return torch.stack([x_base, y_base], dim=1)


def norm_grid(v_grid):
    _, _, height, width = v_grid.size()
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0] = 2.0 * v_grid[:, 0] / (width - 1) - 1.0
    v_grid_norm[:, 1] = 2.0 * v_grid[:, 1] / (height - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)


def flow_warp_2d(x, flow, pad="border", mode="bilinear"):
    batch, _, height, width = x.size()
    base_grid = mesh_grid(batch, height, width).type_as(x)
    sample_grid = norm_grid(base_grid + flow)
    kwargs = {}
    if "align_corners" in inspect.getfullargspec(F.grid_sample).args:
        kwargs["align_corners"] = True
    return F.grid_sample(
        x,
        sample_grid,
        mode=mode,
        padding_mode=pad,
        **kwargs,
    )


def resize_flow_to(flow, height, width):
    _, _, old_height, old_width = flow.shape
    if old_height == height and old_width == width:
        return flow
    resized = F.interpolate(
        flow,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    resized = resized.clone()
    resized[:, 0] *= float(width) / max(float(old_width), 1.0)
    resized[:, 1] *= float(height) / max(float(old_height), 1.0)
    return resized


def resize_feature_to(feature, reference):
    _, _, height, width = reference.shape
    if feature.shape[-2:] == (height, width):
        return feature
    return F.interpolate(
        feature,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )


class LEAConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class LEAUpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class LEAFlowFusion(nn.Module):
    def __init__(self, inputc1, inputc2):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = LEAConvBlock2d(inputc1 + inputc2, 32)
        self.Conv2 = LEAConvBlock2d(32, 64)
        self.Conv3 = LEAConvBlock2d(64, 128)
        self.Conv4 = LEAConvBlock2d(128, 256)
        self.Up4 = LEAUpConv2d(256, 128)
        self.Up_conv4 = LEAConvBlock2d(256, 128)
        self.Up3 = LEAUpConv2d(128, 64)
        self.Up_conv3 = LEAConvBlock2d(128, 64)
        self.Up2 = LEAUpConv2d(64, 32)
        self.Up_conv2 = LEAConvBlock2d(64, 32)
        self.Conv_1x1 = nn.Conv2d(32, inputc2, kernel_size=1, stride=1, padding=0)

    def forward(self, cont1, cont2, flows):
        x = torch.cat([cont1, cont2, flows], dim=1)
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        d4 = self.Up4(x4)
        d4 = resize_feature_to(d4, x3)
        d4 = self.Up_conv4(torch.cat((x3, d4), dim=1))
        d3 = self.Up3(d4)
        d3 = resize_feature_to(d3, x2)
        d3 = self.Up_conv3(torch.cat((x2, d3), dim=1))
        d2 = self.Up2(d3)
        d2 = resize_feature_to(d2, x1)
        d2 = self.Up_conv2(torch.cat((x1, d2), dim=1))
        return self.Conv_1x1(d2)


class LEAContextNet2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.netOne = self.convBlock(1, 8)
        self.netTwo = self.convBlock(8, 16)
        self.netThr = self.convBlock(16, 32)
        self.netFou = self.convBlock(32, 64)

    def convBlock(self, inchannel, outchannel):
        return nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        x_2x = self.netOne(x)
        x_4x = self.netTwo(x_2x)
        x_8x = self.netThr(x_4x)
        x_16x = self.netFou(x_8x)
        return {"1/4": x_4x, "1/8": x_8x, "1/16": x_16x}


class LEAExtractNet2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.netOne = self.convBlock(1, 8)
        self.netTwo = self.convBlock(8, 16)
        self.netThr = self.convBlock(16, 32)
        self.netFou = self.convBlock(32, 64)

    def convBlock(self, inchannel, outchannel):
        return nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        x_2x = self.netOne(x)
        x_4x = self.netTwo(x_2x)
        x_8x = self.netThr(x_4x)
        x_16x = self.netFou(x_8x)
        return {"1/2": x_2x, "1/4": x_4x, "1/8": x_8x, "1/16": x_16x}


class LEAMotionEncoder(nn.Module):
    def __init__(self, inchannel):
        super().__init__()
        self.convc1 = nn.Conv2d(inchannel, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        out = F.relu(self.conv(torch.cat([cor, flo], dim=1)))
        return torch.cat([out, flow], dim=1)


class LEAGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, motion_dim=128):
        super().__init__()
        self.convz = nn.Conv2d(motion_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(motion_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(motion_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class LEAGRUFlowBlock(nn.Module):
    def __init__(self, inchannel, hdim, feature_dim, up=2):
        super().__init__()
        self.encoder = LEAMotionEncoder(inchannel=inchannel)
        self.gru = LEAGRU(input_dim=feature_dim, hidden_dim=feature_dim // 2)
        self.conv1 = nn.Conv2d(hdim, hdim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hdim, 2, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.mask = nn.Sequential(
            nn.Conv2d(feature_dim // 2, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, up * up * 9, 1, padding=0),
        )

    def forward(self, cont, hid, cat_fea, flow):
        motion_features = self.encoder(flow, cat_fea)
        cont = torch.cat([cont, motion_features], dim=1)
        hid = self.gru(hid, cont)
        flow_up = self.conv2(self.lrelu(self.conv1(hid)))
        up_mask = 0.25 * self.mask(hid)
        return flow_up, up_mask


class LEARAFTBlock(nn.Module):
    def __init__(self, inchannel, hdim, feature_dim):
        super().__init__()
        self.GRU_block = LEAGRUFlowBlock(
            inchannel=inchannel,
            hdim=hdim,
            feature_dim=feature_dim,
        )

    def upsample_flow(self, flow, mask):
        batch, _, height, width = flow.shape
        up = int((mask.size(1) // 9) ** 0.5)
        mask = mask.view(batch, 1, 9, up, up, height, width)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(up * flow, [3, 3], padding=1)
        up_flow = up_flow.view(batch, 2, 9, 1, 1, height, width)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(batch, 2, up * height, up * width)

    def forward(self, cont, hid, cat_fea, coords0, coords1, flow=None):
        if flow is not None:
            coords1 = coords1 + flow
        coords1 = coords1.detach()
        flow = coords1 - coords0
        flow_up, up_mask = self.GRU_block(cont, hid, cat_fea, flow)
        return self.upsample_flow(flow_up, up_mask)


class LEALocalFlowNet(nn.Module):
    def __init__(self, flow_multiplier=1.0, use_fusion=True):
        super().__init__()
        self.flow_multiplier = flow_multiplier
        self.use_fusion = use_fusion
        self.fusion = LEAFlowFusion(inputc1=16, inputc2=2)
        self.extraction = LEAExtractNet2d()
        self.baseRaft_16x = LEARAFTBlock(inchannel=128, feature_dim=32 + 32, hdim=32)
        self.baseRaft_8x = LEARAFTBlock(inchannel=64, feature_dim=16 + 16, hdim=16)
        self.baseRaft_4x = LEARAFTBlock(inchannel=32, feature_dim=8 + 8, hdim=8)

    def initialize_flow(self, img):
        batch, _, height, width = img.shape
        coords0 = coords_grid(batch, height, width, device=img.device)
        coords1 = coords_grid(batch, height, width, device=img.device)
        return coords0, coords1

    def forward(self, fixed, moving, fixed_features, context_features, hidden_features):
        moving_features = self.extraction(moving)
        hid_4x, hid_8x, hid_16x = hidden_features
        cont_4x, cont_8x, cont_16x = context_features
        coords0_4, coords1_4 = self.initialize_flow(fixed_features["1/4"])
        coords0_8, coords1_8 = self.initialize_flow(fixed_features["1/8"])
        coords0_16, coords1_16 = self.initialize_flow(fixed_features["1/16"])

        flow_16x_up = self.baseRaft_16x(
            cont_16x,
            hid_16x,
            torch.cat([fixed_features["1/16"], moving_features["1/16"]], dim=1),
            coords0_16,
            coords1_16,
            flow=None,
        )
        flow_16x_up = resize_flow_to(
            flow_16x_up,
            fixed_features["1/8"].shape[-2],
            fixed_features["1/8"].shape[-1],
        )
        flow_8x_up = self.baseRaft_8x(
            cont_8x,
            hid_8x,
            torch.cat([fixed_features["1/8"], moving_features["1/8"]], dim=1),
            coords0_8,
            coords1_8,
            flow=flow_16x_up,
        )
        flow_8x_up = resize_flow_to(
            flow_8x_up,
            fixed_features["1/4"].shape[-2],
            fixed_features["1/4"].shape[-1],
        )
        flow_4x_up = self.baseRaft_4x(
            cont_4x,
            hid_4x,
            torch.cat([fixed_features["1/4"], moving_features["1/4"]], dim=1),
            coords0_4,
            coords1_4,
            flow=flow_8x_up,
        )
        flow_4x_up = resize_flow_to(
            flow_4x_up,
            fixed_features["1/2"].shape[-2],
            fixed_features["1/2"].shape[-1],
        )

        if self.use_fusion:
            flow = self.fusion(fixed_features["1/2"], moving_features["1/2"], flow_4x_up)
        else:
            flow = flow_4x_up
        _, _, height, width = fixed.shape
        final_flow = F.interpolate(
            flow,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ) * 2
        return final_flow * self.flow_multiplier


class LEAExplicitAligner(nn.Module):
    def __init__(self, iters=3, use_fusion=True):
        super().__init__()
        self.args = SimpleNamespace(iters=iters)
        self.flow_multiplier = 1.0 / iters
        self.context = LEAContextNet2d()
        self.extract = LEAExtractNet2d()
        self.defnet = nn.ModuleList(
            [
                LEALocalFlowNet(
                    flow_multiplier=self.flow_multiplier,
                    use_fusion=use_fusion,
                )
                for _ in range(iters)
            ]
        )

    def forward(self, fixed, moving):
        contexts = self.context(fixed)
        fixed_features = self.extract(fixed)
        hid_4x, cont_4x = torch.split(contexts["1/4"], [8, 8], dim=1)
        hid_8x, cont_8x = torch.split(contexts["1/8"], [16, 16], dim=1)
        hid_16x, cont_16x = torch.split(contexts["1/16"], [32, 32], dim=1)
        context_features = [
            torch.relu(cont_4x),
            torch.relu(cont_8x),
            torch.relu(cont_16x),
        ]
        hidden_features = [
            torch.tanh(hid_4x),
            torch.tanh(hid_8x),
            torch.tanh(hid_16x),
        ]

        deform = self.defnet[0](
            fixed,
            moving,
            fixed_features,
            context_features,
            hidden_features,
        )
        warped = flow_warp_2d(moving, deform)
        deforms = [deform]
        warp_images = [warped]
        agg_flow = deform

        for idx in range(self.args.iters - 1):
            deform = self.defnet[idx + 1](
                fixed,
                warped,
                fixed_features,
                context_features,
                hidden_features,
            )
            agg_flow = flow_warp_2d(agg_flow, deform) + deform
            warped = flow_warp_2d(moving, agg_flow)
            deforms.append(deform)
            warp_images.append(warped)

        return deforms, warp_images, warped, agg_flow


# Method-facing name for the local-elastic explicit local module.
# LEA = Local Elastic Aligner.
LEALocalElasticAligner = LEAExplicitAligner


def load_explicit_aligner_weights(model, checkpoint_path, strict=True):
    weights = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(weights, dict) and "state_dict" in weights:
        weights = weights["state_dict"]
    if isinstance(weights, dict) and "model" in weights:
        weights = weights["model"]

    target = model.state_dict()
    if any(key.startswith("local_aligner.") for key in weights):
        weights = {
            key.replace("local_aligner.", "", 1): value
            for key, value in weights.items()
            if key.startswith("local_aligner.")
        }

    missing, unexpected = model.load_state_dict(weights, strict=strict)
    return {"missing": list(missing), "unexpected": list(unexpected)}
