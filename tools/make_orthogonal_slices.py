import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def numeric_pngs(folder):
    files = sorted(Path(folder).glob("*.png"), key=lambda p: int(p.stem))
    if not files:
        raise ValueError(f"No PNG slices found in {folder}")
    return files


def load_xy_stack(xy_dir):
    slices = []
    for path in tqdm(numeric_pngs(xy_dir), desc=f"Loading {xy_dir}"):
        slices.append(np.asarray(Image.open(path).convert("L"), dtype=np.uint8))
    return np.stack(slices, axis=0)


def save_stack(stack, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(stack.shape[0]), desc=f"Saving {out_dir}"):
        array = np.ascontiguousarray(stack[i], dtype=np.uint8)
        height, width = array.shape
        with (out_dir / f"{i}.png").open("wb") as f:
            f.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
            f.write(array.tobytes())


def main(args):
    volume = load_xy_stack(args.xy_dir)
    # volume: [Z, H, W]
    if args.xz_dir:
        # XZ at each y: [H, Z, W]
        save_stack(volume.transpose(1, 0, 2), args.xz_dir)
    if args.yz_dir:
        # YZ at each x: [W, Z, H]
        save_stack(volume.transpose(2, 0, 1), args.yz_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xy_dir", required=True)
    parser.add_argument("--xz_dir", default="")
    parser.add_argument("--yz_dir", default="")
    main(parser.parse_args())
