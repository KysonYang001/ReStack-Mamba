import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from tqdm import tqdm


def numeric_pngs(folder):
    files = sorted(Path(folder).glob("*.png"), key=lambda path: int(path.stem))
    if not files:
        raise ValueError(f"No numeric PNG slices found in {folder}")
    return files


def warp_image(image, dx, dy, x=None, y=None):
    height, width = image.shape
    if x is None or y is None:
        x, y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
    return map_coordinates(
        image,
        (y + dy, x + dx),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).reshape(image.shape).astype(np.uint8)


def main(args):
    files = numeric_pngs(args.input_xy)
    first = np.asarray(Image.open(files[0]).convert("L"), dtype=np.uint8)
    depth = len(files)
    height, width = first.shape
    inverse_flow = np.load(args.reference_inverse_flow, mmap_mode="r")
    expected_shape = (depth, 2, height, width)
    if inverse_flow.shape != expected_shape:
        raise ValueError(
            f"Flow shape {inverse_flow.shape} does not match image stack {expected_shape}"
        )

    alphas = tuple(float(alpha) for alpha in args.alphas)
    output_root = Path(args.output_root)
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    output_dirs = {}
    for alpha in alphas:
        label = str(int(alpha)) if alpha.is_integer() else str(alpha).replace(".", "p")
        output_dir = output_root / f"alpha_{label}" / "simulation" / "XY"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[alpha] = output_dir

    with ThreadPoolExecutor(max_workers=len(alphas)) as executor:
        for z, path in enumerate(tqdm(files, desc="Generate scaled simulations")):
            image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
            if image.shape != (height, width):
                raise ValueError(f"Inconsistent shape at {path}: {image.shape}")
            if z == 0:
                for output_dir in output_dirs.values():
                    Image.fromarray(image).save(output_dir / "0.png", compress_level=1)
                continue

            # The stored inverse field is (-dx, -dy) for reference_alpha.
            base_dx = -np.asarray(inverse_flow[z, 0], dtype=np.float32)
            base_dy = -np.asarray(inverse_flow[z, 1], dtype=np.float32)
            futures = {
                alpha: executor.submit(
                    warp_image,
                    image,
                    base_dx * (alpha / args.reference_alpha),
                    base_dy * (alpha / args.reference_alpha),
                    x,
                    y,
                )
                for alpha in alphas
            }
            for alpha, output_dir in output_dirs.items():
                simulated = futures[alpha].result()
                Image.fromarray(simulated).save(
                    output_dir / f"{z}.png", compress_level=1
                )

    metadata = {
        "input_xy": str(Path(args.input_xy).resolve()),
        "reference_inverse_flow": str(Path(args.reference_inverse_flow).resolve()),
        "shape": [depth, height, width],
        "reference_alpha": args.reference_alpha,
        "alphas": list(alphas),
        "first_slice_identity": True,
        "field_model": "scaled_reference_inverse_flow",
    }
    for alpha, output_dir in output_dirs.items():
        (output_dir.parent / "simulation_metadata.json").write_text(
            json.dumps({**metadata, "alpha": alpha}, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_xy", required=True)
    parser.add_argument("--reference_inverse_flow", required=True)
    parser.add_argument("--reference_alpha", type=float, default=3.0)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--alphas", nargs="+", default=(1, 2, 4, 5))
    main(parser.parse_args())
