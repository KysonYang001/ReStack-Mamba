import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def numeric_pngs(folder):
    files = sorted(Path(folder).glob("*.png"), key=lambda path: int(path.stem))
    if not files:
        raise ValueError(f"No numeric PNG slices found in {folder}")
    return files


def main(args):
    files = numeric_pngs(args.input_dir)
    first = np.asarray(Image.open(files[0]).convert("L"), dtype=np.uint8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    volume = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=np.uint8,
        shape=(len(files), *first.shape),
    )
    for index, path in enumerate(tqdm(files, desc="PNG stack to NPY")):
        image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        if image.shape != first.shape:
            raise ValueError(f"Inconsistent shape at {path}: {image.shape}")
        volume[index] = image
    volume.flush()
    print(f"Saved {output}: shape={volume.shape} dtype={volume.dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
