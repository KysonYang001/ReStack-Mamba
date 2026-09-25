import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def numeric_pngs(folder: str) -> list[Path]:
    files = sorted(Path(folder).glob("*.png"), key=lambda path: int(path.stem))
    indices = [int(path.stem) for path in files]
    if indices != list(range(len(files))):
        raise ValueError(f"Expected contiguous slice indices 0..N-1 in {folder}, got {indices[:3]}..{indices[-3:] if indices else []}")
    return files


def gaussian_window(device: torch.device) -> torch.Tensor:
    coords = torch.arange(11, dtype=torch.float32, device=device) - 5
    kernel = torch.exp(-(coords * coords) / (2 * 1.5 * 1.5))
    kernel /= kernel.sum()
    return (kernel[:, None] @ kernel[None, :]).view(1, 1, 11, 11)


def load_batch(paths: list[Path], device: torch.device) -> torch.Tensor:
    arrays = [np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0 for path in paths]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("Inconsistent image shapes in batch")
    return torch.from_numpy(np.stack(arrays)[:, None]).to(device)


def batch_ssim(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    c1, c2 = 0.01**2, 0.03**2
    mu_x = F.conv2d(x, window, padding=5)
    mu_y = F.conv2d(y, window, padding=5)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, padding=5) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=5) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=5) - mu_xy
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return score.mean(dim=(1, 2, 3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_xy", required=True)
    parser.add_argument("--gt_xy", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--expected_slices", type=int, default=558)
    args = parser.parse_args()

    pred_files = numeric_pngs(args.pred_xy)
    gt_files = numeric_pngs(args.gt_xy)
    if len(pred_files) != args.expected_slices or len(gt_files) != args.expected_slices:
        raise ValueError(f"Expected {args.expected_slices} slices, got pred={len(pred_files)} gt={len(gt_files)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    window = gaussian_window(device)
    rows = []
    with torch.no_grad():
        for start in tqdm(range(0, len(pred_files), args.batch_size), desc="Evaluate XY"):
            stop = min(start + args.batch_size, len(pred_files))
            pred = load_batch(pred_files[start:stop], device)
            gt = load_batch(gt_files[start:stop], device)
            if pred.shape != gt.shape:
                raise ValueError(f"Shape mismatch at batch {start}: {pred.shape} vs {gt.shape}")
            mse = ((pred - gt) ** 2).mean(dim=(1, 2, 3))
            ssim = batch_ssim(pred, gt, window)
            for offset in range(stop - start):
                mse_value = float(mse[offset].item())
                psnr = 100.0 if mse_value == 0.0 else 10.0 * math.log10(1.0 / mse_value)
                row = {"slice": start + offset, "psnr": psnr, "ssim": float(ssim[offset].item())}
                if not np.isfinite([row["psnr"], row["ssim"]]).all():
                    raise ValueError(f"Non-finite metric at slice {row['slice']}: {row}")
                rows.append(row)

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slice", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}", flush=True)


if __name__ == "__main__":
    main()
