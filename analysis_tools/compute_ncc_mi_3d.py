import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image


def numeric_images(directory):
    paths = []
    for path in Path(directory).iterdir():
        if not path.is_file():
            continue
        try:
            index = int(path.name.split(".")[0])
        except ValueError:
            continue
        paths.append((index, path))
    return [path for _, path in sorted(paths)]


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pred_volume")
    source.add_argument("--pred_xy")
    parser.add_argument("--gt_xy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bins", type=int, default=64)
    args = parser.parse_args()

    gt_files = numeric_images(args.gt_xy)
    if args.pred_volume:
        volume = np.load(args.pred_volume, mmap_mode="r")
        count = min(volume.shape[0], len(gt_files))
        pred_files = None
    else:
        pred_files = numeric_images(args.pred_xy)
        count = min(len(pred_files), len(gt_files))
        volume = None

    bins = args.bins
    joint = np.zeros((bins, bins), dtype=np.int64)
    n = 0
    sx = sy = sxx = syy = sxy = 0.0

    for index in range(count):
        pred = np.asarray(volume[index], dtype=np.uint8) if volume is not None else load_gray(pred_files[index])
        gt = load_gray(gt_files[index])
        if pred.shape != gt.shape:
            if pred.T.shape == gt.shape:
                pred = pred.T
            else:
                raise ValueError(f"Slice {index} shape mismatch: pred={pred.shape}, gt={gt.shape}")

        x = pred.reshape(-1).astype(np.float64)
        y = gt.reshape(-1).astype(np.float64)
        n += x.size
        sx += float(x.sum())
        sy += float(y.sum())
        sxx += float(np.dot(x, x))
        syy += float(np.dot(y, y))
        sxy += float(np.dot(x, y))

        xb = (pred.reshape(-1).astype(np.uint16) * bins) // 256
        yb = (gt.reshape(-1).astype(np.uint16) * bins) // 256
        joint += np.bincount(xb * bins + yb, minlength=bins * bins).reshape(bins, bins)
        if (index + 1) % 100 == 0 or index + 1 == count:
            print(f"{index + 1}/{count}", flush=True)

    covariance = sxy - sx * sy / n
    variance_x = sxx - sx * sx / n
    variance_y = syy - sy * sy / n
    ncc = covariance / math.sqrt(max(variance_x * variance_y, 1e-30))

    pxy = joint.astype(np.float64) / joint.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    independent = px[:, None] * py[None, :]
    valid = pxy > 0
    mi = float(np.sum(pxy[valid] * np.log(pxy[valid] / independent[valid])))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slice_count", "voxel_count", "ncc", "mi_nats", "mi_bins"])
        writer.writeheader()
        writer.writerow({
            "slice_count": count,
            "voxel_count": n,
            "ncc": ncc,
            "mi_nats": mi,
            "mi_bins": bins,
        })
    print(f"NCC={ncc:.10f} MI={mi:.10f} nats", flush=True)


if __name__ == "__main__":
    main()
