import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main(args):
    source = np.load(args.input, mmap_mode="r")
    if source.dtype != np.uint8 or source.ndim != 3:
        raise ValueError(f"Expected a 3D uint8 volume, got {source.shape} {source.dtype}")
    radius = max(int(args.radius), 1)
    trim = max(int(args.trim), 0)
    window = 2 * radius + 1
    if 2 * trim >= window:
        raise ValueError(f"trim={trim} removes every value from window={window}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint8,
        shape=source.shape,
    )
    block = max(int(args.y_block), 1)
    for y0 in tqdm(range(0, source.shape[1], block), desc="Trimmed mean volume Z"):
        y1 = min(y0 + block, source.shape[1])
        values = np.asarray(source[:, y0:y1, :], dtype=np.float32)
        padded = np.pad(values, ((radius, radius), (0, 0), (0, 0)), mode="reflect")
        neighbors = np.stack(
            [padded[offset:offset + source.shape[0]] for offset in range(window)],
            axis=0,
        )
        neighbors.sort(axis=0)
        if trim:
            neighbors = neighbors[trim:-trim]
        filtered = neighbors.mean(axis=0)
        output[:, y0:y1, :] = np.clip(np.rint(filtered), 0, 255).astype(np.uint8)
        output.flush()
    print(
        f"Saved {output_path}: shape={output.shape} radius={radius} trim={trim}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--radius", type=int, required=True)
    parser.add_argument("--trim", type=int, required=True)
    parser.add_argument("--y_block", type=int, default=16)
    main(parser.parse_args())
