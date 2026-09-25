#!/usr/bin/env python3
"""Retrain the two released StackMamba stages using checkpoint-recorded settings.

For the shared simulated-volume model, supply paired locally aligned and
undeformed reference NPY volumes for kidney, heart, pancreas, and skin.
Kasthuri11 uses one locally aligned real volume without a reference target.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATASETS = ("jrc_mus_kidney", "jrc_mus_heart", "jrc_mus_pancreas", "jrc_mus_skin")


def run(command, *, env, dry_run):
    print("+", " ".join(map(str, command)), flush=True)
    if not dry_run:
        subprocess.run(list(map(str, command)), cwd=ROOT, env=env, check=True)


def settings_args(settings):
    result = []
    for key, value in settings.items():
        if key == "gpu":
            continue
        flag = "--" + key
        if isinstance(value, bool):
            if value:
                result.append(flag)
        elif isinstance(value, list):
            result.extend((flag, *map(str, value)))
        elif value is not None:
            result.extend((flag, str(value)))
    return result


def main(args):
    specs = {}
    for dataset, source, target in args.dataset:
        if dataset in specs:
            raise ValueError(f"Duplicate dataset: {dataset}")
        source_path = Path(source).resolve()
        target_path = None if target == "-" else Path(target).resolve()
        if not args.dry_run and (not source_path.is_file() or (target_path and not target_path.is_file())):
            raise FileNotFoundError(f"Missing input for {dataset}: {source_path}, {target_path}")
        specs[dataset] = (source_path, target_path)
    if args.checkpoint_set == "shared":
        if set(specs) != set(TRAINING_DATASETS) or any(target is None for _, target in specs.values()):
            raise ValueError(f"Shared model requires paired volumes for {TRAINING_DATASETS}")
        order = TRAINING_DATASETS
    else:
        if set(specs) != {"kasthuri11"} or specs["kasthuri11"][1] is not None:
            raise ValueError("Kasthuri11 training requires: --dataset kasthuri11 ALIGNED.npy -")
        order = ("kasthuri11",)

    metadata = json.loads((ROOT / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    group_metadata = metadata[args.checkpoint_set]
    output = args.output.resolve()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
    sources = [specs[dataset][0] for dataset in order]
    targets = [specs[dataset][1] for dataset in order]

    for stage in (1, 2):
        checkpoint_dir = output / f"stage{stage}"
        command = [sys.executable, "-m", "mamba_reg.train_stack_residual_mamba",
                   "--volume", *sources]
        if args.checkpoint_set == "shared":
            command.extend(["--target_volume", *targets])
        command.extend(["--save_dir", checkpoint_dir, "--gpu", 0])
        command.extend(settings_args(group_metadata[f"stage{stage}"]["training_args"]))
        if not (checkpoint_dir / "best.pth").is_file():
            run(command, env=env, dry_run=args.dry_run)

        if stage == 1:
            checkpoint = checkpoint_dir / "best.pth"
            stage1_sources = []
            for dataset in order:
                stage1_dir = output / "stage1_volumes" / dataset
                stage1_volume = stage1_dir / "registered_volume_uint8.npy"
                if not stage1_volume.is_file():
                    run(
                        [sys.executable, "-m", "mamba_reg.inference_stack_residual_mamba",
                         "--volume", specs[dataset][0], "--checkpoint", checkpoint,
                         "--output_dir", stage1_dir, "--gpu", 0,
                         "--z_size", 16, "--z_stride", 8,
                         "--crop_size", 256, "--xy_stride", 192,
                         "--feature_dim", 48, "--mamba_depth", 4,
                         "--residual_max_flow", 1.0, "--flow_scale", 1.0,
                         "--fusion_mode", "image",
                         "--image_z_residual_smooth_sigma", 0.35, "--remove_tmp"],
                        env=env, dry_run=args.dry_run,
                    )
                stage1_sources.append(stage1_volume)
            sources = stage1_sources


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-set", choices=("shared", "kasthuri11"), default="shared")
    parser.add_argument("--dataset", nargs=3, metavar=("NAME", "ALIGNED_NPY", "REFERENCE_NPY"),
                        action="append", required=True,
                        help="Use - as REFERENCE_NPY for unpaired Kasthuri11")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
