import argparse
import csv
import math
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def numeric_pngs(folder):
    files = sorted(Path(folder).glob("*.png"), key=lambda p: int(p.stem))
    if not files:
        raise ValueError(f"No PNG slices found in {folder}")
    return files


def load_gray(path, device):
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device).view(1, 1, arr.shape[0], arr.shape[1])


def gaussian_window(win_size, sigma, device):
    coords = torch.arange(win_size, dtype=torch.float32, device=device) - win_size // 2
    g = torch.exp(-(coords * coords) / (2 * sigma * sigma))
    g = g / g.sum()
    window = (g[:, None] @ g[None, :]).view(1, 1, win_size, win_size)
    return window


def ssim_2d(x, y, window):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pad = window.shape[-1] // 2
    mu_x = F.conv2d(x, window, padding=pad)
    mu_y = F.conv2d(y, window, padding=pad)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad) - mu_xy
    value = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return value.mean().item()


def psnr_2d(x, y):
    mse = torch.mean((x - y) ** 2).item()
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


def to_lpips_input(x):
    x = x.repeat(1, 3, 1, 1)
    return x * 2.0 - 1.0


def evaluate_pair(pred_dir, gt_dir, out_csv, device, lpips_model, max_slices=0):
    pred_files = numeric_pngs(pred_dir)
    gt_files = numeric_pngs(gt_dir)
    n = min(len(pred_files), len(gt_files))
    if max_slices:
        n = min(n, max_slices)
    window = gaussian_window(11, 1.5, device)
    rows = []
    for i in tqdm(range(n), desc=f"Eval {Path(pred_dir).name}"):
        pred = load_gray(pred_files[i], device)
        gt = load_gray(gt_files[i], device)
        row = {
            "slice": pred_files[i].stem,
            "psnr": psnr_2d(pred, gt),
            "ssim": ssim_2d(pred, gt, window),
            "lpips": lpips_model(to_lpips_input(pred), to_lpips_input(gt)).item(),
        }
        rows.append(row)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["slice", "psnr", "ssim", "lpips"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "count": len(rows),
        "mean_psnr": float(np.mean([r["psnr"] for r in rows])),
        "mean_ssim": float(np.mean([r["ssim"] for r in rows])),
        "mean_lpips": float(np.mean([r["lpips"] for r in rows])),
    }
    return summary


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net=args.lpips_net, pnet_rand=args.lpips_random).to(device)
    loss_fn.eval()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("XY", args.pred_xy, args.gt_xy),
        ("XZ", args.pred_xz, args.gt_xz),
        ("YZ", args.pred_yz, args.gt_yz),
    ]
    summaries = {}
    with torch.no_grad():
        for name, pred, gt in pairs:
            summaries[name] = evaluate_pair(
                pred,
                gt,
                out_dir / f"{name}_metrics.csv",
                device,
                loss_fn,
                args.max_slices,
            )

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["view", "count", "mean_psnr", "mean_ssim", "mean_lpips"],
        )
        writer.writeheader()
        for view, summary in summaries.items():
            writer.writerow({"view": view, **summary})

    for view, summary in summaries.items():
        print(view, summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_xy", required=True)
    parser.add_argument("--pred_xz", required=True)
    parser.add_argument("--pred_yz", required=True)
    parser.add_argument("--gt_xy", required=True)
    parser.add_argument("--gt_xz", required=True)
    parser.add_argument("--gt_yz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lpips_net", default="alex")
    parser.add_argument("--lpips_random", action="store_true")
    parser.add_argument("--max_slices", type=int, default=0)
    main(parser.parse_args())
