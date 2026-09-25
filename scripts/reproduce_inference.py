#!/usr/bin/env python3
"""Run LEA alignment and two-stage StackMamba inference."""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def run(command, *, cwd=ROOT, env=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def check_weights(group):
    expected = {}
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        expected[relative] = digest
    paths = []
    for stage in (1, 2):
        relative = f"checkpoints/{group}/stage{stage}_best.pth"
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing trained weight: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected[relative]:
            raise ValueError(f"Checkpoint checksum mismatch: {relative}")
        paths.append(path)
    return paths


def numeric_slices(directory):
    directory = Path(directory)
    files = [p for p in directory.glob("*.png") if p.stem.isdecimal()]
    files.sort(key=lambda p: int(p.stem))
    if not files or [int(p.stem) for p in files] != list(range(len(files))):
        raise ValueError(f"Expected contiguous PNG slices named 0.png, 1.png, ... in {directory}")
    return files


def check_volume(path, depth):
    volume = np.load(path, mmap_mode="r")
    if volume.ndim != 3 or volume.shape[0] != depth or volume.dtype != np.uint8:
        raise ValueError(f"Unexpected volume at {path}: {volume.shape}, {volume.dtype}")
    return volume


def main(args):
    output = args.output.resolve()
    if args.stage1_checkpoint or args.stage2_checkpoint:
        if not args.stage1_checkpoint or not args.stage2_checkpoint:
            raise ValueError("Pass both --stage1-checkpoint and --stage2-checkpoint")
        checkpoints = [args.stage1_checkpoint.resolve(), args.stage2_checkpoint.resolve()]
        for checkpoint in checkpoints:
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
    else:
        checkpoints = check_weights(args.checkpoint_set)
    input_xy = args.input_xy.resolve()
    input_files = numeric_slices(input_xy)
    depth = len(input_files)
    if args.verify_only:
        if args.aligned_volume:
            check_volume(args.aligned_volume.resolve(), depth)
        elif args.lea_checkpoint and not args.lea_checkpoint.is_file():
            raise FileNotFoundError(args.lea_checkpoint)
        print(f"Verified {depth} input slices and both {args.checkpoint_set} checkpoints.")
        return
    if args.aligned_volume:
        aligned = args.aligned_volume.resolve()
        check_volume(aligned, depth)
    else:
        if not args.lea_checkpoint:
            raise ValueError("Pass --lea-checkpoint or --aligned-volume")
        model = args.lea_checkpoint.resolve()
        if not model.is_file():
            raise FileNotFoundError(model)
        aligned_xy = output / "local_alignment" / "registered_png"
        aligned = output / "local_alignment" / "registered_volume_uint8.npy"
        if not aligned.is_file():
            run(
                [sys.executable, "-m", "mamba_reg.inference_lea_stack",
                 "--input_dir", input_xy, "--output_dir", output / "local_alignment",
                 "--model_path", model, "--gpu", 0, "--iters", 3],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)},
            )
            numeric_slices(aligned_xy)
        check_volume(aligned, depth)

    if args.local_align_only:
        print(f"Local alignment volume: {aligned}")
        return

    current = aligned
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
    for stage, checkpoint in enumerate(checkpoints, start=1):
        stage_dir = output / f"stage{stage}"
        result = stage_dir / "registered_volume_uint8.npy"
        if not result.is_file():
            stage_dir.mkdir(parents=True, exist_ok=True)
            run(
                [sys.executable, "-m", "mamba_reg.inference_stack_residual_mamba",
                 "--volume", current, "--checkpoint", checkpoint,
                 "--output_dir", stage_dir, "--gpu", 0, "--z_size", 16,
                 "--z_stride", 8, "--crop_size", 256, "--xy_stride", 192,
                 "--feature_dim", 48, "--mamba_depth", 4,
                 "--residual_max_flow", 1.0, "--flow_scale", 1.0,
                 "--fusion_mode", "image", "--image_z_residual_smooth_sigma", 0.35,
                 "--remove_tmp"],
                env=env,
            )
        check_volume(result, depth)
        current = result

    for name in ("trim_pass1.npy", "final_volume.npy"):
        target = output / name
        if not target.is_file():
            run([sys.executable, ROOT / "tools" / "trimmed_mean_volume_z.py",
                 "--input", current, "--output", target,
                 "--radius", 2, "--trim", 1, "--y_block", 8])
        check_volume(target, depth)
        current = target

    final_xy = output / "views" / "XY"
    if len(list(final_xy.glob("*.png"))) != depth:
        run([sys.executable, ROOT / "paper_comparison" / "export_volume_xy.py",
             "--volume", current, "--output_dir", final_xy])
    numeric_slices(final_xy)
    for view in ("XZ", "YZ"):
        if not (output / "views" / view).is_dir():
            run([sys.executable, ROOT / "tools" / "make_orthogonal_slices.py",
                 "--xy_dir", final_xy,
                 "--xz_dir", output / "views" / "XZ",
                 "--yz_dir", output / "views" / "YZ"])
            break

    if args.reference_xy:
        reference = args.reference_xy.resolve()
        numeric_slices(reference)
        run([sys.executable, ROOT / "paper_comparison" / "evaluate_xy_metrics.py",
             "--pred_xy", final_xy, "--gt_xy", reference,
             "--output_csv", output / "XY_metrics.csv", "--expected_slices", depth,
             "--device", f"cuda:0"], env=env)
        reference_views = output / "reference_views"
        reference_xz = args.reference_xz.resolve() if args.reference_xz else reference_views / "XZ"
        reference_yz = args.reference_yz.resolve() if args.reference_yz else reference_views / "YZ"
        if not reference_xz.is_dir() or not reference_yz.is_dir():
            run([sys.executable, ROOT / "tools" / "make_orthogonal_slices.py",
                 "--xy_dir", reference, "--xz_dir", reference_xz,
                 "--yz_dir", reference_yz])
        run([sys.executable, "-m", "mamba_reg.evaluate_registered_volume_npy",
             "--pred_volume", current, "--gt_xy", reference,
             "--gt_xz", reference_xz, "--gt_yz", reference_yz,
             "--output_dir", output / "metrics_volume", "--device", "cuda:0",
             "--skip_lpips"], env=env)
        run([sys.executable, ROOT / "analysis_tools" / "compute_ncc_mi_3d.py",
             "--pred_volume", current, "--gt_xy", reference,
             "--output", output / "ncc_mi_3d.csv", "--bins", 64])

    (output / ".complete").touch()
    print(f"Final registered volume: {current}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-xy", required=True, type=Path, help="Contiguous 0.png..N-1.png stack")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-set", choices=("shared", "kasthuri11"), default="shared")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lea-checkpoint", type=Path, help="Trained LEA best.pth")
    parser.add_argument("--stage1-checkpoint", type=Path, help="Stage 1 StackMamba best.pth")
    parser.add_argument("--stage2-checkpoint", type=Path, help="Stage 2 StackMamba best.pth")
    parser.add_argument("--aligned-volume", type=Path, help="Precomputed local alignment in Z,Y,X uint8 NPY")
    parser.add_argument("--reference-xy", type=Path, help="Optional undeformed reference XY slices")
    parser.add_argument("--reference-xz", type=Path, help="Optional precomputed reference XZ slices")
    parser.add_argument("--reference-yz", type=Path, help="Optional precomputed reference YZ slices")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--local-align-only", action="store_true")
    main(parser.parse_args())
