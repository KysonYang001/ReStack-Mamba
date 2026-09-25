import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .loss import GradientLoss, MultiScaleNCC2D
from .stack_residual_mamba import StackResidualMamba


def load_volume(path):
    volume = np.load(path, mmap_mode="r")
    if volume.dtype != np.uint8 and not np.issubdtype(volume.dtype, np.floating):
        raise ValueError(f"Expected uint8 or floating volume, got {volume.dtype}: {path}")
    return volume


def normalize_volume_patch(patch):
    patch = np.asarray(patch)
    was_float = np.issubdtype(patch.dtype, np.floating)
    patch = patch.astype(np.float32, copy=False)
    if was_float and patch.max(initial=0.0) <= 1.5:
        return patch
    return patch / 255.0


class StackPatchDataset(Dataset):
    def __init__(
        self,
        volume_paths,
        target_paths=None,
        target_flow_paths=None,
        z_size=16,
        crop_size=256,
        stride_z=4,
        stride_xy=256,
    ):
        if isinstance(volume_paths, (str, os.PathLike)):
            volume_paths = [volume_paths]
        if target_paths is not None and isinstance(target_paths, (str, os.PathLike)):
            target_paths = [target_paths]
        self.volumes = [load_volume(path) for path in volume_paths]
        self.targets = None
        self.target_flows = None
        if target_paths is not None:
            if len(target_paths) != len(self.volumes):
                raise ValueError("--target_volume must match the number of --volume paths")
            self.targets = [load_volume(path) for path in target_paths]
            for source, target, source_path, target_path in zip(
                self.volumes, self.targets, volume_paths, target_paths
            ):
                if source.shape != target.shape:
                    raise ValueError(
                        f"Paired volume shape mismatch: {source_path}={source.shape}, "
                        f"{target_path}={target.shape}"
                    )
        if target_flow_paths is not None:
            if isinstance(target_flow_paths, (str, os.PathLike)):
                target_flow_paths = [target_flow_paths]
            if len(target_flow_paths) != len(self.volumes):
                raise ValueError("--target_flow must match the number of --volume paths")
            self.target_flows = [np.load(path, mmap_mode="r") for path in target_flow_paths]
            for volume, flow, volume_path, flow_path in zip(
                self.volumes, self.target_flows, volume_paths, target_flow_paths
            ):
                expected = (volume.shape[0], 2, volume.shape[1], volume.shape[2])
                if flow.shape != expected:
                    raise ValueError(
                        f"Target flow shape mismatch: {flow_path}={flow.shape}, "
                        f"expected {expected} for {volume_path}"
                    )
        self.z_size = z_size
        self.crop_size = crop_size
        self.indices = []
        for volume_index, volume in enumerate(self.volumes):
            depth, height, width = volume.shape
            if depth < z_size or height < crop_size or width < crop_size:
                raise ValueError(
                    f"Volume {volume_index} shape {volume.shape} is smaller than "
                    f"patch {(z_size, crop_size, crop_size)}"
                )
            z_starts = list(range(0, depth - z_size + 1, stride_z))
            if z_starts[-1] != depth - z_size:
                z_starts.append(depth - z_size)
            y_starts = list(range(0, height - crop_size + 1, stride_xy))
            if y_starts[-1] != height - crop_size:
                y_starts.append(height - crop_size)
            x_starts = list(range(0, width - crop_size + 1, stride_xy))
            if x_starts[-1] != width - crop_size:
                x_starts.append(width - crop_size)
            for z in z_starts:
                for y in y_starts:
                    for x in x_starts:
                        self.indices.append((volume_index, z, y, x))
        shapes = [tuple(volume.shape) for volume in self.volumes]
        paired = self.targets is not None
        print(
            f"Stack residual dataset: {len(self.indices)} patches from {shapes}, paired={paired}",
            flush=True,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        volume_index, z, y, x = self.indices[idx]
        zs = slice(z, z + self.z_size)
        ys = slice(y, y + self.crop_size)
        xs = slice(x, x + self.crop_size)
        source = normalize_volume_patch(
            np.array(self.volumes[volume_index][zs, ys, xs], copy=True)
        )
        source = torch.from_numpy(source).unsqueeze(0) * 2.0 - 1.0
        if self.targets is None:
            return source
        target = normalize_volume_patch(
            np.array(self.targets[volume_index][zs, ys, xs], copy=True)
        )
        target = torch.from_numpy(target).unsqueeze(0) * 2.0 - 1.0
        if self.target_flows is None:
            return source, target
        target_flow = np.array(
            self.target_flows[volume_index][zs, :, ys, xs],
            dtype=np.float32,
            copy=True,
        )
        target_flow = torch.from_numpy(target_flow).permute(1, 0, 2, 3)
        return source, target, target_flow


def adjacent_ncc_loss(volume, ncc):
    moving = volume[:, :, 1:]
    fixed = volume[:, :, :-1]
    batch, channels, frames, height, width = moving.shape
    moving_2d = moving.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    fixed_2d = fixed.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    return ncc(moving_2d, fixed_2d)


def spaced_ncc_loss(volume, ncc, offset):
    if volume.shape[2] <= offset:
        return volume.new_tensor(0.0)
    moving = volume[:, :, offset:]
    fixed = volume[:, :, :-offset]
    batch, channels, frames, height, width = moving.shape
    moving = moving.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, height, width
    )
    fixed = fixed.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, height, width
    )
    return ncc(moving, fixed)


def paired_ncc_loss(moving, fixed, ncc):
    batch, channels, frames, height, width = moving.shape
    moving = moving.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, height, width
    )
    fixed = fixed.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, height, width
    )
    return ncc(moving, fixed)


def paired_ssim_loss(moving, fixed, window=11):
    batch, channels, frames, height, width = moving.shape
    moving = moving.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, height, width
    )
    fixed = fixed.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, height, width
    )
    window = max(int(window), 3)
    if window % 2 == 0:
        window += 1
    padding = window // 2
    mu_moving = F.avg_pool2d(moving, window, stride=1, padding=padding)
    mu_fixed = F.avg_pool2d(fixed, window, stride=1, padding=padding)
    var_moving = F.avg_pool2d(moving.square(), window, stride=1, padding=padding) - mu_moving.square()
    var_fixed = F.avg_pool2d(fixed.square(), window, stride=1, padding=padding) - mu_fixed.square()
    covariance = F.avg_pool2d(moving * fixed, window, stride=1, padding=padding) - mu_moving * mu_fixed
    c1 = 0.0004
    c2 = 0.0036
    numerator = (2.0 * mu_moving * mu_fixed + c1) * (2.0 * covariance + c2)
    denominator = (mu_moving.square() + mu_fixed.square() + c1) * (
        var_moving + var_fixed + c2
    )
    return 1.0 - (numerator / denominator.clamp_min(1e-6)).mean()


def add_synthetic_jitter(source, stn, max_pixels, grid_size, z_smooth):
    batch, _, frames, height, width = source.shape
    grid_size = max(int(grid_size), 2)
    flow = torch.randn(
        batch,
        2,
        frames,
        grid_size,
        grid_size,
        device=source.device,
        dtype=source.dtype,
    )
    if z_smooth > 0 and frames > 2:
        kernel = min(2 * int(z_smooth) + 1, frames if frames % 2 else frames - 1)
        packed = flow.permute(0, 1, 3, 4, 2).reshape(-1, 1, frames)
        packed = F.avg_pool1d(packed, kernel_size=kernel, stride=1, padding=kernel // 2)
        flow = packed.reshape(batch, 2, grid_size, grid_size, frames).permute(0, 1, 4, 2, 3)
    flow = torch.tanh(flow)
    flow = F.interpolate(
        flow,
        size=(frames, height, width),
        mode="trilinear",
        align_corners=True,
    )
    flow = flow * float(max_pixels)
    return stn(source, flow)


def gradient_magnitude(volume):
    dx = volume[..., :, 1:] - volume[..., :, :-1]
    dy = volume[..., 1:, :] - volume[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-6)


def adjacent_edge_loss(volume):
    edges = gradient_magnitude(volume)
    if edges.shape[2] < 2:
        return edges.new_tensor(0.0)
    difference = edges[:, :, 1:] - edges[:, :, :-1]
    return torch.sqrt(difference.square() + 1e-4).mean()


def image_z_curvature_loss(volume):
    if volume.shape[2] < 3:
        return volume.new_tensor(0.0)
    low_frequency = F.avg_pool3d(
        volume,
        kernel_size=(1, 5, 5),
        stride=1,
        padding=(0, 2, 2),
    )
    ddz = low_frequency[:, :, 2:] - 2.0 * low_frequency[:, :, 1:-1] + low_frequency[:, :, :-2]
    return torch.sqrt(ddz.square() + 1e-6).mean()


def flow_anchor_loss(flow):
    first = flow[:, :, :1].square().mean()
    last = flow[:, :, -1:].square().mean()
    mean = flow.mean(dim=(2, 3, 4)).square().mean()
    return first + last + mean


def flow_z_continuity_loss(flow):
    if flow.shape[2] < 2:
        zero = flow.new_tensor(0.0)
        return zero, zero
    dz = flow[:, :, 1:] - flow[:, :, :-1]
    z_smooth = dz.abs().mean()
    if flow.shape[2] < 3:
        return z_smooth, flow.new_tensor(0.0)
    ddz = flow[:, :, 2:] - 2.0 * flow[:, :, 1:-1] + flow[:, :, :-2]
    z_curvature = ddz.abs().mean()
    return z_smooth, z_curvature


def loss_fn(refined, source, flow, args, ncc, smooth_loss):
    sim = adjacent_ncc_loss(refined, ncc)
    edge_sim = adjacent_edge_loss(refined)
    second_neighbor = spaced_ncc_loss(refined, ncc, offset=2)
    image_z_curvature = image_z_curvature_loss(refined)
    identity = F.mse_loss(refined, source)
    magnitude = flow.square().mean()
    smooth = smooth_loss(flow)
    anchor = flow_anchor_loss(flow)
    z_smooth, z_curvature = flow_z_continuity_loss(flow)
    total = (
        args.sim_weight * sim
        + args.edge_sim_weight * edge_sim
        + args.second_neighbor_weight * second_neighbor
        + args.image_z_curvature_weight * image_z_curvature
        + args.identity_weight * identity
        + args.mag_weight * magnitude
        + args.smooth_weight * smooth
        + args.anchor_weight * anchor
        + args.z_smooth_weight * z_smooth
        + args.z_curvature_weight * z_curvature
    )
    return total, {
        "loss": total,
        "sim": sim,
        "edge_sim": edge_sim,
        "second_neighbor": second_neighbor,
        "image_z_curvature": image_z_curvature,
        "identity": identity,
        "magnitude": magnitude,
        "smooth": smooth,
        "anchor": anchor,
        "z_smooth": z_smooth,
        "z_curvature": z_curvature,
        "flow_abs": flow.abs().mean(),
        "flow_max": flow.abs().amax(),
    }


def save_checkpoint(path, model, optimizer, scaler, epoch, best, args):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best": best,
            "args": vars(args),
            "stack_residual_mamba": True,
        },
        path,
    )


def train_one_epoch(model, loader, optimizer, scaler, device, args, epoch):
    model.train()
    ncc = MultiScaleNCC2D(win=args.ncc_win, scales=tuple(args.ncc_scales)).to(device)
    smooth = GradientLoss().to(device)
    totals = {}
    steps = 0
    skipped_loss = 0
    skipped_grad = 0
    pbar = tqdm(loader, desc=f"stack-residual e{epoch}")
    for step, batch in enumerate(pbar):
        if args.max_steps_per_epoch > 0 and step >= args.max_steps_per_epoch:
            break
        target_flow = None
        if isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                source, target, target_flow = batch
                target_flow = target_flow.to(device, non_blocking=True)
            else:
                source, target = batch
            target = target.to(device, non_blocking=True)
        else:
            source = batch
            target = None
        source = source.to(device, non_blocking=True)
        if target is None:
            target = source
        model_input = source
        if args.synthetic_jitter_max > 0:
            with torch.no_grad():
                model_input = add_synthetic_jitter(
                    source,
                    model.stn,
                    max_pixels=args.synthetic_jitter_max,
                    grid_size=args.synthetic_jitter_grid,
                    z_smooth=args.synthetic_jitter_z_smooth,
                )
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=args.amp and device.type == "cuda"):
            refined, flow = model(model_input)
            loss, parts = loss_fn(refined, model_input, flow, args, ncc, smooth)
            synthetic_ncc = paired_ncc_loss(refined, target, ncc)
            synthetic_mse = F.mse_loss(refined, target)
            paired_l1 = F.l1_loss(refined, target)
            paired_ssim = paired_ssim_loss(refined, target, args.paired_ssim_window)
            if target_flow is None:
                flow_supervision = flow.new_tensor(0.0)
            else:
                flow_supervision = F.smooth_l1_loss(
                    flow,
                    target_flow,
                    beta=args.flow_supervision_beta,
                )
            loss = (
                loss
                + args.synthetic_ncc_weight * synthetic_ncc
                + args.synthetic_mse_weight * synthetic_mse
                + args.paired_l1_weight * paired_l1
                + args.paired_ssim_weight * paired_ssim
                + args.flow_supervision_weight * flow_supervision
            )
            parts["loss"] = loss
            parts["synthetic_ncc"] = synthetic_ncc
            parts["synthetic_mse"] = synthetic_mse
            parts["paired_l1"] = paired_l1
            parts["paired_ssim"] = paired_ssim
            parts["flow_supervision"] = flow_supervision
        if not torch.isfinite(loss):
            skipped_loss += 1
            continue
        if scaler is not None and args.amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm):
            skipped_grad += 1
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and args.amp and device.type == "cuda":
                scaler.update()
            continue
        if scaler is not None and args.amp and device.type == "cuda":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        steps += 1
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item())
        totals["grad_norm"] = totals.get("grad_norm", 0.0) + float(
            grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm
        )
        pbar.set_postfix(
            loss=f"{parts['loss'].detach().item():.4f}",
            sim=f"{parts['sim'].detach().item():.4f}",
            mag=f"{parts['magnitude'].detach().item():.5f}",
            flow=f"{parts['flow_abs'].detach().item():.3f}",
        )
    denom = max(steps, 1)
    for key in list(totals):
        totals[key] /= denom
    totals["steps"] = steps
    totals["skipped_loss"] = skipped_loss
    totals["skipped_grad"] = skipped_grad
    return totals


LOG_FIELDS = [
    "epoch",
    "loss",
    "sim",
    "edge_sim",
    "second_neighbor",
    "image_z_curvature",
    "synthetic_ncc",
    "synthetic_mse",
    "paired_l1",
    "paired_ssim",
    "flow_supervision",
    "identity",
    "magnitude",
    "smooth",
    "anchor",
    "z_smooth",
    "z_curvature",
    "flow_abs",
    "flow_max",
    "grad_norm",
    "steps",
    "skipped_loss",
    "skipped_grad",
]


def append_log(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for field in LOG_FIELDS:
            row.setdefault(field, "")
        writer.writerow(row)


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = StackPatchDataset(
        args.volume,
        target_paths=args.target_volume,
        target_flow_paths=args.target_flow,
        z_size=args.z_size,
        crop_size=args.crop_size,
        stride_z=args.stride_z,
        stride_xy=args.stride_xy,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    model = StackResidualMamba(
        feature_dim=args.feature_dim,
        mamba_depth=args.mamba_depth,
        residual_max_flow=args.residual_max_flow,
        preserve_resolution=args.preserve_resolution,
        preserve_z_resolution=not args.downsample_z,
        bidirectional_mamba=args.bidirectional_mamba,
    ).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state, strict=True)
        print(f"Initialized model from {args.init_checkpoint}", flush=True)
    if args.reset_flow_head:
        torch.nn.init.normal_(model.decoder.flow_head.weight, mean=0.0, std=1e-5)
        if model.decoder.flow_head.bias is not None:
            torch.nn.init.constant_(model.decoder.flow_head.bias, 0.0)
        print("Reset decoder flow head after initialization", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best = float("inf")
    for epoch in range(args.epochs):
        stats = train_one_epoch(model, loader, optimizer, scaler, device, args, epoch)
        print(f"[stack_residual Epoch {epoch}] {stats}", flush=True)
        row = {"epoch": epoch, **stats}
        append_log(Path(args.save_dir) / "train_log.csv", row)
        save_checkpoint(Path(args.save_dir) / "latest.pth", model, optimizer, scaler, epoch, best, args)
        if stats.get("loss", float("inf")) < best:
            best = stats["loss"]
            save_checkpoint(Path(args.save_dir) / "best.pth", model, optimizer, scaler, epoch, best, args)
            print(f"Best model saved: {best:.6f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", required=True, nargs="+")
    parser.add_argument("--target_volume", nargs="+")
    parser.add_argument("--target_flow", nargs="+")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--init_checkpoint")
    parser.add_argument("--reset_flow_head", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--z_size", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--stride_z", type=int, default=4)
    parser.add_argument("--stride_xy", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max_steps_per_epoch", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--feature_dim", type=int, default=24)
    parser.add_argument("--mamba_depth", type=int, default=2)
    parser.add_argument("--bidirectional_mamba", action="store_true")
    parser.add_argument("--residual_max_flow", type=float, default=1.5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--ncc_win", type=int, default=9)
    parser.add_argument("--ncc_scales", type=float, nargs="+", default=[1.0, 0.5])
    parser.add_argument("--sim_weight", type=float, default=1.0)
    parser.add_argument("--edge_sim_weight", type=float, default=0.0)
    parser.add_argument("--second_neighbor_weight", type=float, default=0.0)
    parser.add_argument("--image_z_curvature_weight", type=float, default=0.0)
    parser.add_argument("--synthetic_jitter_max", type=float, default=0.0)
    parser.add_argument("--synthetic_jitter_grid", type=int, default=8)
    parser.add_argument("--synthetic_jitter_z_smooth", type=int, default=1)
    parser.add_argument("--synthetic_ncc_weight", type=float, default=0.0)
    parser.add_argument("--synthetic_mse_weight", type=float, default=0.0)
    parser.add_argument("--paired_l1_weight", type=float, default=0.0)
    parser.add_argument("--paired_ssim_weight", type=float, default=0.0)
    parser.add_argument("--paired_ssim_window", type=int, default=11)
    parser.add_argument("--flow_supervision_weight", type=float, default=0.0)
    parser.add_argument("--flow_supervision_beta", type=float, default=1.0)
    parser.add_argument("--identity_weight", type=float, default=0.25)
    parser.add_argument("--mag_weight", type=float, default=0.08)
    parser.add_argument("--smooth_weight", type=float, default=0.20)
    parser.add_argument("--anchor_weight", type=float, default=0.05)
    parser.add_argument("--z_smooth_weight", type=float, default=0.0)
    parser.add_argument("--z_curvature_weight", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--preserve_resolution", action="store_true")
    parser.add_argument(
        "--downsample_z",
        action="store_true",
        help="Use the legacy isotropic encoder that downsamples Z together with XY.",
    )
    parser.add_argument("--amp", action="store_true")
    main(parser.parse_args())
