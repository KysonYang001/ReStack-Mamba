import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    volume = np.load(args.volume, mmap_mode="r")
    if volume.ndim != 3 or volume.dtype != np.uint8:
        raise ValueError(f"Expected 3D uint8 volume, got {volume.shape} {volume.dtype}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for index in tqdm(range(volume.shape[0]), desc="Export XY"):
        Image.fromarray(np.asarray(volume[index])).save(output / f"{index:04d}.png")


if __name__ == "__main__":
    main()
