#!/usr/bin/env python3
"""Download and reproduce the six OpenOrganelle simulation inputs."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "jrc_mus_liver": ("jrc_mus-liver", 558, 4.0),
    "jrc_mus_kidney": ("jrc_mus-kidney", 1387, 3.0),
    "jrc_mus_heart": ("jrc_mus-heart-1", 1061, 3.0),
    "jrc_mus_pancreas": ("jrc_mus-pancreas-4", 898, 3.0),
    "jrc_mus_skin": ("jrc_mus-skin-1", 1231, 3.0),
    "jrc_mus_liver3": ("jrc_mus-liver-3", 1127, 3.0),
}


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def count_slices(directory):
    files = [p for p in directory.glob("*.png") if p.stem.isdecimal()]
    if files and sorted(int(p.stem) for p in files) != list(range(len(files))):
        raise ValueError(f"Non-contiguous slice names: {directory}")
    return len(files)


def main(args):
    for slug in args.datasets:
        dataset_id, expected, alpha = DATASETS[slug]
        base = args.data_root.resolve() / slug
        reference = base / "orignal" / "XY"  # original server spelling
        simulated = base / "simulated" / "XY"
        if count_slices(reference) != expected:
            run([sys.executable, ROOT / "tools" / "download_openorganelle_s4.py",
                 "--dataset", dataset_id, "--scale", "s4", "--size", 800,
                 "--output_dir", reference])
        if count_slices(reference) != expected:
            raise ValueError(f"Incomplete downloaded dataset: {slug}")
        reference_volume = base / "orignal" / "volume_uint8.npy"
        if not reference_volume.is_file():
            run([sys.executable, ROOT / "tools" / "png_stack_to_npy.py",
                 "--input_dir", reference, "--output", reference_volume])
        if count_slices(simulated) != expected:
            run([sys.executable, ROOT / "tools" / "generate_elastic_simulation.py",
                 "--input_xy", reference, "--output_xy", simulated,
                 "--inverse_flow", base / "inverse_flow_xy_float16.npy",
                 "--seed", 42, "--alpha", alpha, "--sigma", 0.08])
        if count_slices(simulated) != expected:
            raise ValueError(f"Incomplete simulation: {slug}")
        print(f"READY {slug}: {expected} slices, alpha={alpha}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--datasets", choices=tuple(DATASETS), nargs="+", default=list(DATASETS))
    main(parser.parse_args())
