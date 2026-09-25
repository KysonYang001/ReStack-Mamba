import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from tqdm import tqdm


def numeric_pngs(folder):
    files = sorted(Path(folder).glob("*.png"), key=lambda path: int(path.stem))
    if not files:
        raise ValueError(f"No numeric PNG slices found in {folder}")
    return files


def elastic_field(shape, random_state, alpha, sigma):
    height, width = shape
    dx = gaussian_filter(
        random_state.rand(height, width) * 2.0 - 1.0,
        sigma=width * sigma,
    ) * (width * alpha)
    dy = gaussian_filter(
        random_state.rand(height, width) * 2.0 - 1.0,
        sigma=width * sigma,
    ) * (width * alpha)
    return dx.astype(np.float32), dy.astype(np.float32)


def warp_image(image, dx, dy):
    height, width = image.shape
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
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
    output_dir = Path(args.output_xy)
    output_dir.mkdir(parents=True, exist_ok=True)
    flow_path = Path(args.inverse_flow)
    flow_path.parent.mkdir(parents=True, exist_ok=True)
    inverse_flow = np.lib.format.open_memmap(
        flow_path,
        mode="w+",
        dtype=np.float16,
        shape=(depth, 2, height, width),
    )
    random_state = np.random.RandomState(args.seed)

    for z, path in enumerate(tqdm(files, desc="Generate elastic simulation")):
        image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        if image.shape != (height, width):
            raise ValueError(f"Inconsistent shape at {path}: {image.shape}")
        if z == 0:
            simulated = image
            inverse_flow[z] = 0
        else:
            dx, dy = elastic_field(image.shape, random_state, args.alpha, args.sigma)
            simulated = warp_image(image, dx, dy)
            inverse_flow[z, 0] = (-dx).astype(np.float16)
            inverse_flow[z, 1] = (-dy).astype(np.float16)
        Image.fromarray(simulated).save(output_dir / f"{z}.png")
        if z % 16 == 0:
            inverse_flow.flush()
    inverse_flow.flush()

    metadata = {
        "input_xy": str(Path(args.input_xy).resolve()),
        "shape": [depth, height, width],
        "seed": args.seed,
        "alpha": args.alpha,
        "sigma": args.sigma,
        "first_slice_identity": True,
        "field_model": "independent_2d_gaussian_filtered_uniform_noise",
    }
    (output_dir.parent / "simulation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_xy", required=True)
    parser.add_argument("--output_xy", required=True)
    parser.add_argument("--inverse_flow", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--sigma", type=float, default=0.08)
    main(parser.parse_args())
