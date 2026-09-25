import argparse
import csv
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from .coarse_edge_aligner import (
    add_coarse_alignment_args,
    coarse_alignment_kwargs,
    prepare_coarse_aligned_folder,
)
from .dataset import load_volume_from_folder
from .lea_local_aligner import LEALocalElasticAligner, load_explicit_aligner_weights


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def pearson_similarity_loss(img1, img2_warped):
    sizes = img1[0].numel()
    flatten1 = img1.reshape(-1, sizes)
    flatten2 = img2_warped.reshape(-1, sizes)
    mean1 = torch.mean(flatten1, dim=-1, keepdim=True)
    mean2 = torch.mean(flatten2, dim=-1, keepdim=True)
    var1 = torch.mean((flatten1 - mean1) ** 2, dim=-1)
    var2 = torch.mean((flatten2 - mean2) ** 2, dim=-1)
    cov12 = torch.mean((flatten1 - mean1) * (flatten2 - mean2), dim=-1)
    pearson_r = cov12 / torch.sqrt((var1 + 1e-6) * (var2 + 1e-6))
    return torch.sum(1.0 - pearson_r)


def regularize_loss_2d(flow):
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    return ((dx * dx).mean() + (dy * dy).mean()) / 2.0


class LEATripletPatchDataset(Dataset):
    def __init__(
        self,
        volume_dir,
        crop_size=256,
        stride_z=1,
        stride_xy=256,
        resize_size=0,
        normalize=True,
    ):
        self.volume = load_volume_from_folder(volume_dir, mmap_mode="r")
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.normalize = normalize
        depth, height, width = self.volume.shape
        self.indices = []
        for z in range(1, depth - 1, stride_z):
            for y in range(0, height - crop_size + 1, stride_xy):
                for x in range(0, width - crop_size + 1, stride_xy):
                    self.indices.append((z, y, x))
        print(f"LEA triplet dataset: {len(self.indices)} patches from {self.volume.shape}", flush=True)

    def __len__(self):
        return len(self.indices)

    def _slice(self, z, y, x):
        c = self.crop_size
        arr = np.array(self.volume[z, y:y + c, x:x + c], dtype=np.float32, copy=True)
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).float().view(1, 1, c, c)
        if self.resize_size > 0 and arr.shape != (self.resize_size, self.resize_size):
            tensor = F.interpolate(
                tensor,
                size=(self.resize_size, self.resize_size),
                mode="bilinear",
                align_corners=False,
            )
        tensor = tensor.squeeze(0)
        if self.normalize:
            tensor = tensor * 2.0 - 1.0
        return tensor

    def __getitem__(self, idx):
        z, y, x = self.indices[idx]
        return {
            "forward": self._slice(z - 1, y, x),
            "moving": self._slice(z, y, x),
            "backward": self._slice(z + 1, y, x),
        }


def lea_triplet_loss(deforms, warped_center, prev_img, next_img, theta):
    sim = pearson_similarity_loss(prev_img, warped_center) + pearson_similarity_loss(next_img, warped_center)
    reg = warped_center.new_tensor(0.0)
    for deform in deforms:
        reg = reg + regularize_loss_2d(deform)
    reg = reg * 10.0
    return sim + theta * reg, {"sim": sim, "reg": reg}


def save_checkpoint(path, model, optimizer, scheduler, epoch, best, args):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best": best,
            "args": vars(args),
            "module_name": "LEA_LocalElasticAligner",
        },
        path,
    )


def train_one_epoch(model, loader, optimizer, device, args, epoch):
    model.train()
    totals = {"loss": 0.0, "sim": 0.0, "reg": 0.0, "grad_norm": 0.0, "steps": 0}
    pbar = tqdm(loader, desc=f"LEA e{epoch}")
    for step, batch in enumerate(pbar):
        if args.max_steps_per_epoch > 0 and step >= args.max_steps_per_epoch:
            break
        prev_img = batch["forward"].to(device, non_blocking=True)
        center = batch["moving"].to(device, non_blocking=True)
        next_img = batch["backward"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        deforms, _, warped_center, _ = model(prev_img, center)
        loss, parts = lea_triplet_loss(deforms, warped_center, prev_img, next_img, theta=args.theta)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()
        totals["loss"] += float(loss.detach().item())
        totals["sim"] += float(parts["sim"].detach().item())
        totals["reg"] += float(parts["reg"].detach().item())
        totals["grad_norm"] += float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
        totals["steps"] += 1
        pbar.set_postfix(
            loss=f"{loss.detach().item():.4f}",
            sim=f"{parts['sim'].detach().item():.4f}",
            reg=f"{parts['reg'].detach().item():.6f}",
        )
    denom = max(totals["steps"], 1)
    for key in ["loss", "sim", "reg", "grad_norm"]:
        totals[key] /= denom
    return totals


@torch.no_grad()
def validate_one_epoch(model, loader, device, args, epoch):
    model.eval()
    totals = {"loss": 0.0, "sim": 0.0, "reg": 0.0, "steps": 0}
    pbar = tqdm(loader, desc=f"LEA val e{epoch}")
    for batch in pbar:
        prev_img = batch["forward"].to(device, non_blocking=True)
        center = batch["moving"].to(device, non_blocking=True)
        next_img = batch["backward"].to(device, non_blocking=True)
        deforms, _, warped_center, _ = model(prev_img, center)
        loss, parts = lea_triplet_loss(deforms, warped_center, prev_img, next_img, theta=args.theta)
        if not torch.isfinite(loss):
            continue
        totals["loss"] += float(loss.item())
        totals["sim"] += float(parts["sim"].item())
        totals["reg"] += float(parts["reg"].item())
        totals["steps"] += 1
    denom = max(totals["steps"], 1)
    for key in ["loss", "sim", "reg"]:
        totals[key] /= denom
    return totals


def append_log(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        "loss",
        "sim",
        "reg",
        "grad_norm",
        "steps",
        "val_loss",
        "val_sim",
        "val_reg",
        "val_steps",
        "lr",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main(args):
    init_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    volume_dir = args.volume_dir
    if args.coarse_align:
        volume_dir = str(
            prepare_coarse_aligned_folder(
                args.volume_dir,
                Path(args.save_dir) / "coarse_aligned_input",
                **coarse_alignment_kwargs(args),
            )
        )
        print(f"LEA training uses edge-coarse-aligned input: {volume_dir}", flush=True)
    dataset = LEATripletPatchDataset(
        volume_dir,
        crop_size=args.crop_size,
        stride_z=args.stride_z,
        stride_xy=args.stride_xy,
        resize_size=args.resize_size,
        normalize=not args.no_normalize,
    )
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val_ratio must be between 0 and 1")
    val_count = max(1, int(round(len(dataset) * args.val_ratio)))
    train_count = len(dataset) - val_count
    train_dataset, val_dataset = random_split(
        dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(
        f"Official LEA split: train={len(train_dataset)} val={len(val_dataset)} "
        f"resize={args.resize_size}",
        flush=True,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    model = LEALocalElasticAligner(iters=args.iters).to(device)
    if args.init_checkpoint:
        status = load_explicit_aligner_weights(model, args.init_checkpoint, strict=not args.relaxed_init)
        print(f"Initialized LEA from {args.init_checkpoint}: {status}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    milestone = max(int(0.9 * args.epochs), 1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone], gamma=0.1)
    best = float("inf")
    for epoch in range(args.epochs):
        stats = train_one_epoch(model, loader, optimizer, device, args, epoch)
        val_stats = validate_one_epoch(model, val_loader, device, args, epoch)
        stats["epoch"] = epoch
        stats["lr"] = optimizer.param_groups[0]["lr"]
        stats["val_loss"] = val_stats["loss"]
        stats["val_sim"] = val_stats["sim"]
        stats["val_reg"] = val_stats["reg"]
        stats["val_steps"] = val_stats["steps"]
        print(f"[LEA Epoch {epoch}] {stats}", flush=True)
        append_log(Path(args.save_dir) / "train_log.csv", stats)
        save_checkpoint(Path(args.save_dir) / "latest.pth", model, optimizer, scheduler, epoch, best, args)
        if val_stats["loss"] < best and val_stats["steps"] > 0:
            best = val_stats["loss"]
            save_checkpoint(Path(args.save_dir) / "best.pth", model, optimizer, scheduler, epoch, best, args)
            print(f"Best LEA validation model saved: {best:.6f}", flush=True)
        if args.save_each_epoch:
            save_checkpoint(Path(args.save_dir) / f"{epoch + 1}.pth", model, optimizer, scheduler, epoch, best, args)
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--relaxed_init", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--stride_z", type=int, default=1)
    parser.add_argument("--stride_xy", type=int, default=256)
    parser.add_argument("--resize_size", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_steps_per_epoch", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--theta", type=float, default=18.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--save_each_epoch", action="store_true")
    add_coarse_alignment_args(parser)
    main(parser.parse_args())
