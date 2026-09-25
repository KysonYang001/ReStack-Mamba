import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _numeric_key(path):
    numbers = re.findall(r"\d+", Path(path).name)
    return int(numbers[0]) if numbers else 0


def load_uint8_stack(folder):
    files = _image_files(folder)
    if not files:
        raise ValueError(f"No images found in {folder}")
    volume = np.stack(
        [np.asarray(Image.open(path).convert("L"), dtype=np.uint8) for path in files],
        axis=0,
    )
    return files, volume


def _image_files(folder):
    return sorted(
        (path for path in Path(folder).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=_numeric_key,
    )


def _source_signature(files):
    return [
        {"name": path.name, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in files
    ]


def edge_map(image, sigma=1.5):
    image = np.asarray(image, dtype=np.float32)
    if sigma > 0:
        image = ndimage.gaussian_filter(image, sigma=float(sigma), mode="reflect")
    gx = ndimage.sobel(image, axis=1, mode="reflect")
    gy = ndimage.sobel(image, axis=0, mode="reflect")
    magnitude = np.hypot(gx, gy)
    positive = magnitude[magnitude > 0]
    if positive.size == 0:
        return np.zeros_like(magnitude, dtype=np.float32)
    scale = float(np.percentile(positive, 90.0))
    return np.clip(magnitude / max(scale, 1e-6), 0.0, 1.0).astype(np.float32)


def _resize_for_search(image, max_side):
    height, width = image.shape
    scale = min(1.0, float(max_side) / float(max(height, width)))
    if scale >= 1.0:
        return np.asarray(image, dtype=np.float32), 1.0
    resized = ndimage.zoom(image, zoom=scale, order=1, mode="reflect", prefilter=False)
    return np.asarray(resized, dtype=np.float32), scale


def _phase_correlation_shift(fixed, moving, max_shift):
    fixed = np.asarray(fixed, dtype=np.float32)
    moving = np.asarray(moving, dtype=np.float32)
    window_y = np.hanning(fixed.shape[0]).astype(np.float32)
    window_x = np.hanning(fixed.shape[1]).astype(np.float32)
    window = window_y[:, None] * window_x[None, :]
    fixed_fft = np.fft.fft2((fixed - fixed.mean()) * window)
    moving_fft = np.fft.fft2((moving - moving.mean()) * window)
    cross_power = fixed_fft * np.conj(moving_fft)
    cross_power /= np.maximum(np.abs(cross_power), 1e-8)
    correlation = np.abs(np.fft.fftshift(np.fft.ifft2(cross_power)))
    center = np.asarray(correlation.shape) // 2
    radius = int(max_shift)
    if radius > 0:
        y0 = max(int(center[0] - radius), 0)
        y1 = min(int(center[0] + radius + 1), correlation.shape[0])
        x0 = max(int(center[1] - radius), 0)
        x1 = min(int(center[1] + radius + 1), correlation.shape[1])
        local = correlation[y0:y1, x0:x1]
        local_peak = np.unravel_index(int(np.argmax(local)), local.shape)
        peak = np.asarray([local_peak[0] + y0, local_peak[1] + x0])
    else:
        peak = np.asarray(np.unravel_index(int(np.argmax(correlation)), correlation.shape))
    shift = peak.astype(np.float32) - center.astype(np.float32)
    return float(shift[0]), float(shift[1])


def _overlap_score(fixed, moving, shift_y, shift_x):
    shifted = ndimage.shift(
        moving,
        shift=(shift_y, shift_x),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    valid = ndimage.shift(
        np.ones_like(moving, dtype=np.float32),
        shift=(shift_y, shift_x),
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ) > 0.5
    margin_y = max(int(round(fixed.shape[0] * 0.03)), 1)
    margin_x = max(int(round(fixed.shape[1] * 0.03)), 1)
    valid[:margin_y] = False
    valid[-margin_y:] = False
    valid[:, :margin_x] = False
    valid[:, -margin_x:] = False
    if valid.sum() < 64:
        return -1.0
    a = fixed[valid]
    b = shifted[valid]
    denom = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / max(denom, 1e-8))


def estimate_edge_rigid_transform(
    fixed,
    moving,
    max_rotation=30.0,
    coarse_angle_step=2.0,
    fine_angle_step=0.25,
    max_translation=0.25,
    edge_sigma=1.5,
    search_max_side=512,
):
    fixed_small, scale = _resize_for_search(fixed, search_max_side)
    moving_small, _ = _resize_for_search(moving, search_max_side)
    fixed_edge = edge_map(fixed_small, sigma=edge_sigma * scale)
    moving_edge = edge_map(moving_small, sigma=edge_sigma * scale)
    max_shift = int(round(float(max_translation) * max(fixed_edge.shape)))
    if max_translation >= 1.0:
        max_shift = int(round(float(max_translation) * scale))

    def evaluate(angle):
        rotated = ndimage.rotate(
            moving_edge,
            angle=float(angle),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        shift_y, shift_x = _phase_correlation_shift(fixed_edge, rotated, max_shift)
        score = _overlap_score(fixed_edge, rotated, shift_y, shift_x)
        return score, shift_y, shift_x

    coarse_step = max(float(coarse_angle_step), 1e-3)
    coarse_angles = np.arange(-max_rotation, max_rotation + 0.5 * coarse_step, coarse_step)
    best = max(((*evaluate(angle), float(angle)) for angle in coarse_angles), key=lambda item: item[0])
    best_score, best_y, best_x, best_angle = best

    fine_step = max(float(fine_angle_step), 1e-3)
    fine_angles = np.arange(best_angle - coarse_step, best_angle + coarse_step + 0.5 * fine_step, fine_step)
    fine_angles = np.clip(fine_angles, -max_rotation, max_rotation)
    best = max(((*evaluate(angle), float(angle)) for angle in fine_angles), key=lambda item: item[0])
    best_score, best_y, best_x, best_angle = best
    return {
        "angle_deg": float(best_angle),
        "shift_y": float(best_y / scale),
        "shift_x": float(best_x / scale),
        "score": float(best_score),
    }


def apply_rigid_transform(image, transform):
    fill = float(np.median(image))
    rotated = ndimage.rotate(
        image,
        angle=transform["angle_deg"],
        reshape=False,
        order=1,
        mode="constant",
        cval=fill,
        prefilter=False,
    )
    aligned = ndimage.shift(
        rotated,
        shift=(transform["shift_y"], transform["shift_x"]),
        order=1,
        mode="constant",
        cval=fill,
        prefilter=False,
    )
    return np.clip(aligned, 0, 255).astype(np.uint8)


def align_stack_by_edges(volume, anchor=-1, min_score=0.15, **kwargs):
    volume = np.asarray(volume, dtype=np.uint8)
    depth = volume.shape[0]
    anchor = depth // 2 if anchor < 0 else int(np.clip(anchor, 0, depth - 1))
    aligned = np.empty_like(volume)
    aligned[anchor] = volume[anchor]
    transforms = [None] * depth
    transforms[anchor] = {
        "slice": anchor,
        "reference": anchor,
        "angle_deg": 0.0,
        "shift_y": 0.0,
        "shift_x": 0.0,
        "score": 1.0,
        "accepted": True,
    }

    directions = [range(anchor + 1, depth), range(anchor - 1, -1, -1)]
    for indices, label in zip(directions, ["forward", "backward"]):
        for index in tqdm(indices, desc=f"Edge coarse alignment {label}"):
            reference = index - 1 if index > anchor else index + 1
            transform = estimate_edge_rigid_transform(aligned[reference], volume[index], **kwargs)
            accepted = transform["score"] >= float(min_score)
            if not accepted:
                transform = {
                    "angle_deg": 0.0,
                    "shift_y": 0.0,
                    "shift_x": 0.0,
                    "score": transform["score"],
                }
            aligned[index] = apply_rigid_transform(volume[index], transform)
            transforms[index] = {
                "slice": index,
                "reference": reference,
                **transform,
                "accepted": accepted,
            }
    return aligned, transforms


def prepare_coarse_aligned_folder(input_dir, output_dir, force=False, **kwargs):
    output_dir = Path(output_dir)
    metadata_path = output_dir / "coarse_alignment.json"
    input_files = _image_files(input_dir)
    if not input_files:
        raise ValueError(f"No images found in {input_dir}")
    source_signature = _source_signature(input_files)
    if metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        png_count = len(list(output_dir.glob("*.png")))
        if (
            metadata.get("parameters") == kwargs
            and metadata.get("source_signature") == source_signature
            and png_count == metadata.get("slice_count")
        ):
            print(f"Reusing coarse-aligned stack: {output_dir}", flush=True)
            return output_dir

    files, volume = load_uint8_stack(input_dir)
    aligned, transforms = align_stack_by_edges(volume, **kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_path in output_dir.glob("*.png"):
        old_path.unlink()
    for index, image in enumerate(tqdm(aligned, desc="Save coarse-aligned PNG")):
        Image.fromarray(image).save(output_dir / f"{index}.png")
    with (output_dir / "coarse_transforms.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "slice",
                "reference",
                "angle_deg",
                "shift_y",
                "shift_x",
                "score",
                "accepted",
            ],
        )
        writer.writeheader()
        writer.writerows(transforms)
    metadata = {
        "input_dir": str(Path(input_dir).resolve()),
        "slice_count": len(files),
        "parameters": kwargs,
        "source_signature": source_signature,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


def coarse_alignment_kwargs(args):
    return {
        "anchor": args.coarse_anchor,
        "max_rotation": args.coarse_max_rotation,
        "coarse_angle_step": args.coarse_angle_step,
        "fine_angle_step": args.coarse_fine_angle_step,
        "max_translation": args.coarse_max_translation,
        "edge_sigma": args.coarse_edge_sigma,
        "search_max_side": args.coarse_search_max_side,
        "min_score": args.coarse_min_score,
    }


def add_coarse_alignment_args(parser):
    parser.add_argument("--coarse_align", action="store_true", help="Enable edge-based rigid pre-alignment.")
    parser.add_argument("--coarse_anchor", type=int, default=-1, help="Anchor slice; -1 selects the middle slice.")
    parser.add_argument("--coarse_max_rotation", type=float, default=30.0)
    parser.add_argument("--coarse_angle_step", type=float, default=2.0)
    parser.add_argument("--coarse_fine_angle_step", type=float, default=0.25)
    parser.add_argument(
        "--coarse_max_translation",
        type=float,
        default=0.25,
        help="Maximum translation: image-size fraction when <1, otherwise pixels.",
    )
    parser.add_argument("--coarse_edge_sigma", type=float, default=1.5)
    parser.add_argument("--coarse_search_max_side", type=int, default=512)
    parser.add_argument("--coarse_min_score", type=float, default=0.15)


def main(args):
    kwargs = coarse_alignment_kwargs(args)
    prepare_coarse_aligned_folder(args.input_dir, args.output_dir, force=args.force, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--force", action="store_true")
    add_coarse_alignment_args(parser)
    main(parser.parse_args())
