import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


S3_HTTP_ROOT = "https://janelia-cosem-datasets.s3.amazonaws.com"


def orient_and_resize(image, size):
    image = np.asarray(image, dtype=np.uint8)
    image = np.ascontiguousarray(image.T)
    return np.asarray(
        Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def open_remote_n5(dataset, scale):
    try:
        import tensorstore as ts
    except ImportError as exc:
        raise RuntimeError(
            "tensorstore is required; install tensorstore==0.1.78 for Python 3.10"
        ) from exc
    base_url = f"{S3_HTTP_ROOT}/{dataset}/{dataset}.n5/em/fibsem-uint8/{scale}"
    store = ts.open(
        {
            "driver": "n5",
            "kvstore": {"driver": "http", "base_url": base_url},
            "context": {
                "cache_pool": {"total_bytes_limit": 512 * 1024 * 1024},
                "data_copy_concurrency": {"limit": 8},
                "http_request_concurrency": {"limit": 32},
            },
        },
        read=True,
    ).result()
    return store, base_url


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store, source_url = open_remote_n5(args.dataset, args.scale)
    shape = tuple(int(value) for value in store.shape)
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D N5 volume, got {shape}")
    depth = shape[2] if args.max_slices <= 0 else min(shape[2], args.max_slices)
    print(
        f"OpenOrganelle {args.dataset}/{args.scale}: source_shape={shape} "
        f"output_shape=({depth}, {args.size}, {args.size})",
        flush=True,
    )

    progress = tqdm(total=depth, desc=f"Download {args.dataset}/{args.scale}")
    for z0 in range(0, depth, args.z_chunk):
        z1 = min(z0 + args.z_chunk, depth)
        paths = [output_dir / f"{z}.png" for z in range(z0, z1)]
        if not args.force and all(path.exists() for path in paths):
            progress.update(z1 - z0)
            continue
        block = np.asarray(store[:, :, z0:z1].read().result(), dtype=np.uint8)
        for local_z, output_path in enumerate(paths):
            if output_path.exists() and not args.force:
                progress.update(1)
                continue
            image = orient_and_resize(block[:, :, local_z], args.size)
            Image.fromarray(image).save(output_path)
            progress.update(1)
    progress.close()

    metadata = {
        "dataset": args.dataset,
        "scale": args.scale,
        "source_url": source_url,
        "source_shape": shape,
        "output_shape": [depth, args.size, args.size],
        "spatial_transform": "transpose_then_bilinear_resize",
    }
    (output_dir.parent / "download_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scale", default="s4")
    parser.add_argument("--size", type=int, default=800)
    parser.add_argument("--max_slices", type=int, default=0)
    parser.add_argument("--z_chunk", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args())
