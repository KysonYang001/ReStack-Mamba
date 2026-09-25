import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .stack_residual_mamba import StackResidualMamba


def starts_for(length, size, stride):
    if size >= length:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    last = length - size
    if starts[-1] != last:
        starts.append(last)
    return starts


def gaussian_1d(length, sigma_scale=0.25):
    if length <= 1:
        return np.ones((length,), dtype=np.float32)
    coords = np.arange(length, dtype=np.float32)
    center = (length - 1) / 2.0
    sigma = max(length * sigma_scale, 1.0)
    return np.exp(-0.5 * ((coords - center) / sigma) ** 2).astype(np.float32)


def patch_weight(depth, height, width):
    wz = gaussian_1d(depth, sigma_scale=0.35)
    wy = gaussian_1d(height, sigma_scale=0.25)
    wx = gaussian_1d(width, sigma_scale=0.25)
    return np.maximum(wz[:, None, None] * wy[None, :, None] * wx[None, None, :], 1e-4)


def gaussian_kernel_1d(sigma, radius=0):
    if sigma <= 0:
        return np.asarray([1.0], dtype=np.float32)
    if radius <= 0:
        radius = max(int(round(3.0 * sigma)), 1)
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (coords / float(sigma)) ** 2)
    return (kernel / kernel.sum()).astype(np.float32)


def smooth_flow_along_z(flow, sigma, y_block=64, radius=0):
    kernel = gaussian_kernel_1d(sigma, radius=radius)
    if len(kernel) == 1:
        return
    pad = len(kernel) // 2
    _, depth, height, _ = flow.shape
    y_block = max(int(y_block), 1)
    print(
        f"Smoothing fused flow along z: sigma={sigma} radius={pad} y_block={y_block}",
        flush=True,
    )
    for channel in range(flow.shape[0]):
        for y0 in tqdm(range(0, height, y_block), desc=f"Smooth flow c{channel}"):
            y1 = min(y0 + y_block, height)
            block = np.asarray(flow[channel, :, y0:y1, :], dtype=np.float32)
            padded = np.pad(block, ((pad, pad), (0, 0), (0, 0)), mode="reflect")
            smoothed = np.zeros_like(block)
            for k, weight in enumerate(kernel):
                smoothed += float(weight) * padded[k:k + depth]
            flow[channel, :, y0:y1, :] = smoothed
        flow.flush()


def median_filter_flow_along_z(flow, radius, y_block=64):
    radius = int(radius)
    if radius <= 0:
        return
    _, depth, height, _ = flow.shape
    y_block = max(int(y_block), 1)
    window = 2 * radius + 1
    print(
        f"Median filtering fused flow along z: radius={radius} window={window} y_block={y_block}",
        flush=True,
    )
    for channel in range(flow.shape[0]):
        for y0 in tqdm(range(0, height, y_block), desc=f"Median flow c{channel}"):
            y1 = min(y0 + y_block, height)
            block = np.asarray(flow[channel, :, y0:y1, :], dtype=np.float32)
            padded = np.pad(block, ((radius, radius), (0, 0), (0, 0)), mode="reflect")
            windows = [padded[k:k + depth] for k in range(window)]
            flow[channel, :, y0:y1, :] = np.median(np.stack(windows, axis=0), axis=0)
        flow.flush()


def smooth_flow_curvature_along_z(flow, weight, iterations, y_block=64):
    weight = float(weight)
    iterations = int(iterations)
    if weight <= 0.0 or iterations <= 0:
        return
    _, depth, height, _ = flow.shape
    if depth < 3:
        return
    y_block = max(int(y_block), 1)
    print(
        f"Curvature smoothing fused flow along z: weight={weight} iterations={iterations} "
        f"y_block={y_block}",
        flush=True,
    )
    for channel in range(flow.shape[0]):
        for y0 in tqdm(range(0, height, y_block), desc=f"Curvature flow c{channel}"):
            y1 = min(y0 + y_block, height)
            block = np.asarray(flow[channel, :, y0:y1, :], dtype=np.float32).copy()
            for _ in range(iterations):
                updated = block.copy()
                neighbor_avg = 0.5 * (block[:-2] + block[2:])
                updated[1:-1] = (block[1:-1] + weight * neighbor_avg) / (1.0 + weight)
                block = updated
            flow[channel, :, y0:y1, :] = block
        flow.flush()


def to_tensor(patch, device):
    tensor = torch.from_numpy(np.ascontiguousarray(patch, dtype=np.float32)).to(device)
    tensor = tensor.view(1, 1, patch.shape[0], patch.shape[1], patch.shape[2]) / 255.0
    return tensor * 2.0 - 1.0


def save_pngs(volume, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for z in tqdm(range(volume.shape[0]), desc="Save PNG"):
        Image.fromarray(volume[z]).save(out_dir / f"{z}.png")


def smooth_registered_residual_along_z(registered, base_volume, sigma, y_block=32, radius=0):
    kernel = gaussian_kernel_1d(sigma, radius=radius)
    if len(kernel) == 1:
        return
    pad = len(kernel) // 2
    depth, height, _ = registered.shape
    y_block = max(int(y_block), 1)
    print(
        f"Smoothing registered residual along z: sigma={sigma} radius={pad} y_block={y_block}",
        flush=True,
    )
    for y0 in tqdm(range(0, height, y_block), desc="Smooth image residual"):
        y1 = min(y0 + y_block, height)
        base_block = np.asarray(base_volume[:, y0:y1, :], dtype=np.float32)
        residual = np.asarray(registered[:, y0:y1, :], dtype=np.float32) - base_block
        padded = np.pad(residual, ((pad, pad), (0, 0), (0, 0)), mode="reflect")
        smoothed = np.zeros_like(residual)
        for k, weight in enumerate(kernel):
            smoothed += float(weight) * padded[k:k + depth]
        registered[:, y0:y1, :] = np.clip(base_block + smoothed, 0, 255).astype(np.uint8)
        registered.flush()


@torch.no_grad()
def infer_image_fusion(model, volume, out_dir, args, device):
    depth, height, width = volume.shape
    z_starts = starts_for(depth, args.z_size, args.z_stride)
    y_starts = starts_for(height, args.crop_size, args.xy_stride)
    x_starts = starts_for(width, args.crop_size, args.xy_stride)
    accum = np.lib.format.open_memmap(
        out_dir / "_accum_float32.npy",
        mode="w+",
        dtype=np.float32,
        shape=(depth, height, width),
    )
    weights = np.lib.format.open_memmap(
        out_dir / "_weights_float32.npy",
        mode="w+",
        dtype=np.float32,
        shape=(depth, height, width),
    )
    accum[:] = 0.0
    weights[:] = 0.0
    total = len(z_starts) * len(y_starts) * len(x_starts)
    print(
        f"Stack residual image-fusion inference: volume={volume.shape} tiles={len(z_starts)}x"
        f"{len(y_starts)}x{len(x_starts)} total={total} flow_scale={args.flow_scale} "
        f"image_z_residual_smooth_sigma={args.image_z_residual_smooth_sigma}",
        flush=True,
    )
    for z0 in tqdm(z_starts, desc="Infer stack residual z"):
        z1 = min(z0 + args.z_size, depth)
        z0 = max(0, z1 - args.z_size)
        for y0 in y_starts:
            y1 = min(y0 + args.crop_size, height)
            y0 = max(0, y1 - args.crop_size)
            for x0 in x_starts:
                x1 = min(x0 + args.crop_size, width)
                x0 = max(0, x1 - args.crop_size)
                patch = np.array(volume[z0:z1, y0:y1, x0:x1], dtype=np.float32, copy=True)
                tensor = to_tensor(patch, device)
                flow = model.predict_flow(tensor) * args.flow_scale
                refined = model.stn(tensor, flow)
                refined_np = (
                    refined.squeeze().detach().cpu().numpy().clip(-1.0, 1.0) * 0.5 + 0.5
                ) * 255.0
                w = patch_weight(z1 - z0, y1 - y0, x1 - x0)
                accum[z0:z1, y0:y1, x0:x1] += refined_np.astype(np.float32) * w
                weights[z0:z1, y0:y1, x0:x1] += w
        accum.flush()
        weights.flush()

    refined_u8 = np.lib.format.open_memmap(
        out_dir / "registered_volume_uint8.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(depth, height, width),
    )
    for z in tqdm(range(depth), desc="Finalize volume"):
        refined = accum[z] / np.maximum(weights[z], 1e-6)
        refined_u8[z] = np.clip(refined, 0, 255).astype(np.uint8)
    refined_u8.flush()
    smooth_registered_residual_along_z(
        refined_u8,
        volume,
        sigma=args.image_z_residual_smooth_sigma,
        y_block=args.image_z_smooth_y_block,
        radius=args.image_z_smooth_radius,
    )
    return refined_u8


@torch.no_grad()
def warp_volume_with_flow(model, volume, flow, out_path, device, z_chunk):
    depth, height, width = volume.shape
    refined_u8 = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.uint8,
        shape=(depth, height, width),
    )
    z_chunk = max(int(z_chunk), 1)
    for z0 in tqdm(range(0, depth, z_chunk), desc="Warp smoothed flow"):
        z1 = min(z0 + z_chunk, depth)
        patch = np.asarray(volume[z0:z1], dtype=np.float32)
        tensor = to_tensor(patch, device)
        flow_tensor = torch.from_numpy(
            np.ascontiguousarray(flow[:, z0:z1], dtype=np.float32)
        ).to(device).view(1, 2, z1 - z0, height, width)
        refined = model.stn(tensor, flow_tensor)
        refined_np = (
            refined.squeeze().detach().cpu().numpy().clip(-1.0, 1.0) * 0.5 + 0.5
        ) * 255.0
        refined_u8[z0:z1] = np.clip(refined_np, 0, 255).astype(np.uint8)
        refined_u8.flush()
    return refined_u8


def main(args):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    feature_dim = int(args.feature_dim or ckpt_args.get("feature_dim", 24))
    mamba_depth = int(args.mamba_depth or ckpt_args.get("mamba_depth", 2))
    residual_max_flow = float(args.residual_max_flow or ckpt_args.get("residual_max_flow", 1.5))
    preserve_resolution = bool(args.preserve_resolution or ckpt_args.get("preserve_resolution", False))
    bidirectional_mamba = bool(
        args.bidirectional_mamba or ckpt_args.get("bidirectional_mamba", False)
    )
    downsample_z = bool(args.downsample_z or ckpt_args.get("downsample_z", False))

    model = StackResidualMamba(
        feature_dim=feature_dim,
        mamba_depth=mamba_depth,
        residual_max_flow=residual_max_flow,
        preserve_resolution=preserve_resolution,
        preserve_z_resolution=not downsample_z,
        bidirectional_mamba=bidirectional_mamba,
    ).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    volume = np.load(args.volume, mmap_mode="r")
    depth, height, width = volume.shape
    z_starts = starts_for(depth, args.z_size, args.z_stride)
    y_starts = starts_for(height, args.crop_size, args.xy_stride)
    x_starts = starts_for(width, args.crop_size, args.xy_stride)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.fusion_mode == "image":
        refined_u8 = infer_image_fusion(model, volume, out_dir, args, device)
        if args.save_png:
            save_pngs(refined_u8, out_dir / "registered_png")
        if args.remove_tmp:
            for path in [out_dir / "_accum_float32.npy", out_dir / "_weights_float32.npy"]:
                try:
                    os.remove(path)
                except OSError:
                    pass
        print(f"Saved {out_dir / 'registered_volume_uint8.npy'}", flush=True)
        return

    flow_accum = np.lib.format.open_memmap(
        out_dir / "_flow_accum_float32.npy",
        mode="w+",
        dtype=np.float32,
        shape=(2, depth, height, width),
    )
    weights = np.lib.format.open_memmap(
        out_dir / "_weights_float32.npy",
        mode="w+",
        dtype=np.float32,
        shape=(depth, height, width),
    )
    flow_accum[:] = 0.0
    weights[:] = 0.0
    total = len(z_starts) * len(y_starts) * len(x_starts)
    print(
        f"Stack residual inference: volume={volume.shape} tiles={len(z_starts)}x"
        f"{len(y_starts)}x{len(x_starts)} total={total} "
        f"flow_scale={args.flow_scale} "
        f"flow_z_median_radius={args.flow_z_median_radius} "
        f"flow_z_curvature_weight={args.flow_z_curvature_weight} "
        f"flow_z_curvature_iters={args.flow_z_curvature_iters} "
        f"flow_z_smooth_sigma={args.flow_z_smooth_sigma}",
        flush=True,
    )
    with torch.no_grad():
        for z0 in tqdm(z_starts, desc="Infer stack residual z"):
            z1 = min(z0 + args.z_size, depth)
            z0 = max(0, z1 - args.z_size)
            for y0 in y_starts:
                y1 = min(y0 + args.crop_size, height)
                y0 = max(0, y1 - args.crop_size)
                for x0 in x_starts:
                    x1 = min(x0 + args.crop_size, width)
                    x0 = max(0, x1 - args.crop_size)
                    patch = np.array(volume[z0:z1, y0:y1, x0:x1], dtype=np.float32, copy=True)
                    tensor = to_tensor(patch, device)
                    flow = model.predict_flow(tensor) * args.flow_scale
                    flow_np = flow.squeeze(0).detach().cpu().numpy().astype(np.float32)
                    w = patch_weight(z1 - z0, y1 - y0, x1 - x0)
                    flow_accum[:, z0:z1, y0:y1, x0:x1] += flow_np * w[None]
                    weights[z0:z1, y0:y1, x0:x1] += w
            flow_accum.flush()
            weights.flush()

    fused_flow = np.lib.format.open_memmap(
        out_dir / "_fused_flow_float32.npy",
        mode="w+",
        dtype=np.float32,
        shape=(2, depth, height, width),
    )
    for channel in range(2):
        for z in tqdm(range(depth), desc=f"Fuse flow c{channel}"):
            fused_flow[channel, z] = flow_accum[channel, z] / np.maximum(weights[z], 1e-6)
        fused_flow.flush()

    median_filter_flow_along_z(
        fused_flow,
        radius=args.flow_z_median_radius,
        y_block=args.flow_z_smooth_y_block,
    )
    smooth_flow_curvature_along_z(
        fused_flow,
        weight=args.flow_z_curvature_weight,
        iterations=args.flow_z_curvature_iters,
        y_block=args.flow_z_smooth_y_block,
    )
    smooth_flow_along_z(
        fused_flow,
        sigma=args.flow_z_smooth_sigma,
        y_block=args.flow_z_smooth_y_block,
        radius=args.flow_z_smooth_radius,
    )
    refined_u8 = warp_volume_with_flow(
        model,
        volume,
        fused_flow,
        out_dir / "registered_volume_uint8.npy",
        device,
        args.warp_z_chunk,
    )
    if args.save_png:
        save_pngs(refined_u8, out_dir / "registered_png")
    if args.remove_tmp:
        for path in [
            out_dir / "_flow_accum_float32.npy",
            out_dir / "_weights_float32.npy",
            out_dir / "_fused_flow_float32.npy",
        ]:
            try:
                os.remove(path)
            except OSError:
                pass
    print(f"Saved {out_dir / 'registered_volume_uint8.npy'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--z_size", type=int, default=16)
    parser.add_argument("--z_stride", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--xy_stride", type=int, default=192)
    parser.add_argument("--feature_dim", type=int, default=0)
    parser.add_argument("--mamba_depth", type=int, default=0)
    parser.add_argument("--residual_max_flow", type=float, default=0.0)
    parser.add_argument("--preserve_resolution", action="store_true")
    parser.add_argument("--bidirectional_mamba", action="store_true")
    parser.add_argument(
        "--downsample_z",
        action="store_true",
        help="Use the legacy isotropic encoder that downsamples Z together with XY.",
    )
    parser.add_argument("--flow_scale", type=float, default=1.0)
    parser.add_argument("--fusion_mode", choices=["image", "flow"], default="image")
    parser.add_argument("--image_z_residual_smooth_sigma", type=float, default=0.35)
    parser.add_argument("--image_z_smooth_radius", type=int, default=0)
    parser.add_argument("--image_z_smooth_y_block", type=int, default=32)
    parser.add_argument("--flow_z_median_radius", type=int, default=0)
    parser.add_argument("--flow_z_curvature_weight", type=float, default=0.0)
    parser.add_argument("--flow_z_curvature_iters", type=int, default=0)
    parser.add_argument("--flow_z_smooth_sigma", type=float, default=0.0)
    parser.add_argument("--flow_z_smooth_radius", type=int, default=0)
    parser.add_argument("--flow_z_smooth_y_block", type=int, default=64)
    parser.add_argument("--warp_z_chunk", type=int, default=16)
    parser.add_argument("--save_png", action="store_true")
    parser.add_argument("--remove_tmp", action="store_true")
    main(parser.parse_args())
