import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .coarse_edge_aligner import (
    add_coarse_alignment_args,
    coarse_alignment_kwargs,
    prepare_coarse_aligned_folder,
)
from .lea_local_aligner import LEALocalElasticAligner, flow_warp_2d, load_explicit_aligner_weights


def sort_by_number(path):
    numbers = re.findall(r"\d+", Path(path).name)
    return int(numbers[0]) if numbers else 0


def numeric_images(folder):
    folder = Path(folder)
    files = [
        path
        for path in folder.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    ]
    return sorted(files, key=sort_by_number)


def load_uint8_stack(folder):
    files = numeric_images(folder)
    if not files:
        raise ValueError(f"No images found in {folder}")
    volume = [np.asarray(Image.open(path).convert("L"), dtype=np.uint8) for path in files]
    return files, volume


def save_uint8_png(array, path):
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def gaussian_weights(sigma, radius):
    size = 2 * radius + 1
    center = (size - 1) / 2.0
    x = np.arange(size, dtype=np.float32) - center
    weights = np.exp(-0.5 * (x / float(sigma)) ** 2)
    return weights / np.sum(weights)


def preprocess_uint8(image, device):
    tensor = torch.from_numpy(np.ascontiguousarray(image)).to(device=device, dtype=torch.float32)
    tensor = tensor.view(1, 1, image.shape[0], image.shape[1]) / 255.0
    return tensor * 2.0 - 1.0


def postprocess_uint8(tensor):
    image = (tensor.detach().clamp(-1.0, 1.0).squeeze().cpu().numpy() * 0.5 + 0.5) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


@torch.no_grad()
def flow_estimation(moving, fixed, model, device):
    fixed_tensor = preprocess_uint8(fixed, device)
    moving_tensor = preprocess_uint8(moving, device)
    _, _, _, flow = model(fixed_tensor, moving_tensor)
    return flow


@torch.no_grad()
def img_update(image, flow, device):
    image_tensor = preprocess_uint8(image, device)
    warped = flow_warp_2d(image_tensor, flow)
    return postprocess_uint8(warped)


def padded_sequence(seq, radius):
    padded = [img.copy() for img in seq]
    for _ in range(radius - 1):
        padded.insert(0, seq[0].copy())
    for _ in range(radius):
        padded.insert(-1, seq[-1].copy())
    return padded


@torch.no_grad()
def run_lea_pass(seq, names, model, device, radius, layer_range, sigma, desc):
    if len(seq) < 2:
        return [img.copy() for img in seq]
    height, width = seq[0].shape
    weights = gaussian_weights(sigma, radius)
    img_paths = padded_sequence(seq, radius)
    out_seq = [img.copy() for img in seq]
    out_seq[0] = seq[0].copy()

    start_i = radius
    end_i = len(seq) + radius - 1
    imgs_cur = []
    progress = tqdm(range(start_i, end_i), desc=desc)
    for i in progress:
        imgs_tmp = []
        ws_tmp = []
        if not imgs_cur:
            imgs_cur = [img_paths[i + k].copy() for k in range(-radius, radius + 1)]
        else:
            imgs_cur = imgs_cur[1:]
            imgs_cur.append(img_paths[i + radius].copy())

        target_idx = len(imgs_cur) // 2
        for k in range(-radius + 1, radius + 1):
            w_tmp = 0.0
            phi_sum = torch.zeros((1, 2, height, width), dtype=torch.float32, device=device)
            for l in range(max(-radius, k - layer_range), k):
                denom = k - max(-radius, k - layer_range)
                w_tmp += weights[radius + l] + weights[radius + k] / float(denom)
                phi = weights[l + radius] * flow_estimation(
                    moving=imgs_cur[k + radius],
                    fixed=imgs_cur[l + radius],
                    model=model,
                    device=device,
                )
                phi_sum = phi_sum + phi

            img_tmp = img_update(imgs_cur[target_idx + k], phi_sum, device)
            ws_tmp.append(w_tmp)
            imgs_tmp.append(img_tmp)
            imgs_cur[target_idx + k] = img_tmp

        ws_tmp = np.asarray(ws_tmp, dtype=np.float32)
        ws_tmp = ws_tmp / max(float(np.sum(ws_tmp)), 1e-8)

        phi_1 = flow_estimation(
            moving=imgs_cur[target_idx],
            fixed=imgs_cur[target_idx - 1],
            model=model,
            device=device,
        )
        phi_2 = torch.zeros((1, 2, height, width), dtype=torch.float32, device=device)
        for weight, img_tmp in zip(ws_tmp, imgs_tmp):
            phi_tmp = flow_estimation(
                moving=imgs_cur[target_idx],
                fixed=img_tmp,
                model=model,
                device=device,
            )
            phi_2 = phi_2 + float(weight) * phi_tmp

        final_flow = (phi_1 + phi_2) / 2.0
        warped = img_update(imgs_cur[target_idx], final_flow, device)
        imgs_cur[target_idx] = warped

        out_idx = i - radius + 1
        out_seq[out_idx] = warped
        img_paths[i] = warped
        if names:
            progress.set_postfix_str(Path(names[out_idx]).name)

    return out_seq


def main(args):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    input_dir = args.input_dir
    if args.coarse_align:
        input_dir = str(
            prepare_coarse_aligned_folder(
                args.input_dir,
                Path(args.output_dir) / "coarse_aligned_input",
                **coarse_alignment_kwargs(args),
            )
        )
        print(f"LEA inference uses edge-coarse-aligned input: {input_dir}", flush=True)
    files, seq = load_uint8_stack(input_dir)
    if args.max_slices > 0:
        files = files[: args.max_slices]
        seq = seq[: args.max_slices]
    names = [path.name for path in files]
    height, width = seq[0].shape

    model = LEALocalElasticAligner(iters=args.iters).to(device)
    load_info = load_explicit_aligner_weights(model, args.model_path, strict=True)
    print(f"Loaded LEA local elastic aligner weights: {load_info}", flush=True)
    model.eval()

    print(
        f"LEA stack inference: slices={len(seq)} size={height}x{width} "
        f"times={args.times} r={args.r} L={args.L} sigma={args.sigma}",
        flush=True,
    )
    current = seq
    for pass_idx in range(args.times):
        reverse = bool(pass_idx % 2)
        if reverse:
            order = list(range(len(current) - 1, -1, -1))
            ordered_seq = [current[idx] for idx in order]
            ordered_names = [names[idx] for idx in order]
            ordered_out = run_lea_pass(
                ordered_seq,
                ordered_names,
                model,
                device,
                args.r,
                args.L,
                args.sigma,
                desc=f"Pass {pass_idx + 1}/{args.times} reverse",
            )
            next_seq = [None] * len(current)
            for img, idx in zip(ordered_out, order):
                next_seq[idx] = img
            current = next_seq
        else:
            current = run_lea_pass(
                current,
                names,
                model,
                device,
                args.r,
                args.L,
                args.sigma,
                desc=f"Pass {pass_idx + 1}/{args.times} forward",
            )

    out_dir = Path(args.output_dir)
    png_dir = out_dir / "registered_png"
    png_dir.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(tqdm(current, desc="Save PNG")):
        save_uint8_png(image, png_dir / names[idx])

    volume = np.stack(current, axis=0).astype(np.uint8)
    np.save(out_dir / "registered_volume_uint8.npy", volume)
    print(f"Saved {out_dir / 'registered_volume_uint8.npy'} shape={volume.shape}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--times", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--max_slices", type=int, default=0)
    add_coarse_alignment_args(parser)
    main(parser.parse_args())
