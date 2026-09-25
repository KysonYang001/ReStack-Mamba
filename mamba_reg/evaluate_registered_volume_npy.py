import argparse
import csv
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .evaluate_stack_metrics import (
    gaussian_window,
    numeric_pngs,
    psnr_2d,
    ssim_2d,
    to_lpips_input,
)


def load_gray_float(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def as_tensor(array, device):
    array = np.ascontiguousarray(array, dtype=np.float32)
    return torch.from_numpy(array).to(device).view(1, 1, array.shape[0], array.shape[1])


def evaluate_view(volume, gt_dir, view, out_csv, device, lpips_model):
    gt_files = numeric_pngs(gt_dir)
    if view == "XY":
        count = min(volume.shape[0], len(gt_files))
        get_pred = lambda i: volume[i]
    elif view == "XZ":
        count = min(volume.shape[1], len(gt_files))
        get_pred = lambda i: volume[:, i, :]
    elif view == "YZ":
        count = min(volume.shape[2], len(gt_files))
        get_pred = lambda i: volume[:, :, i]
    else:
        raise ValueError(view)

    window = gaussian_window(11, 1.5, device)
    rows = []
    for i in tqdm(range(count), desc=f"Eval {view}"):
        pred_np = get_pred(i).astype(np.float32) / 255.0
        gt_np = load_gray_float(gt_files[i])
        if pred_np.shape != gt_np.shape:
            if pred_np.T.shape == gt_np.shape:
                pred_np = pred_np.T
            else:
                raise ValueError(f"{view} slice {i} shape mismatch: pred={pred_np.shape}, gt={gt_np.shape}")
        pred = as_tensor(pred_np, device)
        gt = as_tensor(gt_np, device)
        rows.append(
            {
                "slice": str(i),
                "psnr": psnr_2d(pred, gt),
                "ssim": ssim_2d(pred, gt, window),
                "lpips": (
                    lpips_model(to_lpips_input(pred), to_lpips_input(gt)).item()
                    if lpips_model is not None
                    else float("nan")
                ),
            }
        )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["slice", "psnr", "ssim", "lpips"])
        writer.writeheader()
        writer.writerows(rows)

    return {
        "count": len(rows),
        "mean_psnr": float(np.mean([float(r["psnr"]) for r in rows])),
        "mean_ssim": float(np.mean([float(r["ssim"]) for r in rows])),
        "mean_lpips": float(np.mean([float(r["lpips"]) for r in rows])),
    }


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    volume = np.load(args.pred_volume, mmap_mode="r")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lpips_model = None
    if not args.skip_lpips:
        lpips_model = lpips.LPIPS(net=args.lpips_net, pnet_rand=args.lpips_random).to(device).eval()
    summaries = {}
    with torch.no_grad():
        summaries["XY"] = evaluate_view(volume, args.gt_xy, "XY", out_dir / "XY_metrics.csv", device, lpips_model)
        summaries["XZ"] = evaluate_view(volume, args.gt_xz, "XZ", out_dir / "XZ_metrics.csv", device, lpips_model)
        summaries["YZ"] = evaluate_view(volume, args.gt_yz, "YZ", out_dir / "YZ_metrics.csv", device, lpips_model)

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["view", "count", "mean_psnr", "mean_ssim", "mean_lpips"])
        writer.writeheader()
        for view, summary in summaries.items():
            writer.writerow({"view": view, **summary})
            print(view, summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_volume", required=True)
    parser.add_argument("--gt_xy", required=True)
    parser.add_argument("--gt_xz", required=True)
    parser.add_argument("--gt_yz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lpips_net", default="alex")
    parser.add_argument("--lpips_random", action="store_true")
    parser.add_argument("--skip_lpips", action="store_true")
    main(parser.parse_args())
