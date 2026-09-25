# dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


############################################################
# Load Volume
############################################################

def load_volume_from_folder(folder, mmap_mode=None):
    """
    folder:
        0.png
        1.png
        2.png
        ...

    return:
        volume [F,H,W]
    """

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Volume folder does not exist: {folder}")

    cache_path = os.path.join(folder, ".volume_float32.npy")
    if os.path.exists(cache_path):
        volume = np.load(cache_path, mmap_mode=mmap_mode)
        print(f"Loaded volume cache: {cache_path} shape={volume.shape}", flush=True)
        return volume

    def numeric_key(name):
        stem = os.path.splitext(name)[0]
        try:
            return int(stem)
        except ValueError as exc:
            raise ValueError(
                f"Slice file names must be numeric, got: {name}"
            ) from exc

    files = sorted(
        [f for f in os.listdir(folder)
         if f.lower().endswith(".png")],
        key=numeric_key
    )

    if not files:
        raise ValueError(f"No PNG slices found in: {folder}")

    print(f"Building volume cache from PNG slices: {folder}", flush=True)
    first_path = os.path.join(folder, files[0])
    with Image.open(first_path) as first_img:
        first_arr = np.asarray(first_img.convert("L"), dtype=np.float32) / 255.0

    volume = np.empty((len(files), first_arr.shape[0], first_arr.shape[1]), dtype=np.float32)
    volume[0] = first_arr

    for idx, fname in enumerate(files[1:], start=1):
        with Image.open(os.path.join(folder, fname)) as img:
            volume[idx] = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    tmp_path = cache_path + ".tmp"
    np.save(tmp_path, volume)
    os.replace(tmp_path + ".npy", cache_path)
    print(f"Saved volume cache: {cache_path} shape={volume.shape}", flush=True)

    return volume


############################################################
# Training Dataset
############################################################

class VEMTrainDataset(Dataset):

    """
    Output:

    [1,Z,H,W]

    Example:

    [1,16,256,256]
    """

    def __init__(
        self,
        volume_dir,
        window_size=16,
        crop_size=32,
        stride_z=8,
        stride_xy=16
    ):

        super().__init__()

        self.window_size = window_size
        self.crop_size = crop_size

        self.volume = load_volume_from_folder(
            volume_dir
        )

        self.depth = self.volume.shape[0]
        self.height = self.volume.shape[1]
        self.width = self.volume.shape[2]

        self.indices = []

        ####################################################
        # build index table
        ####################################################

        for z in range(
            0,
            self.depth - window_size + 1,
            stride_z
        ):

            for y in range(
                0,
                self.height - crop_size + 1,
                stride_xy
            ):

                for x in range(
                    0,
                    self.width - crop_size + 1,
                    stride_xy
                ):

                    self.indices.append(
                        (z, y, x)
                    )

        print(
            f"Dataset initialized: "
            f"{len(self.indices)} patches"
        )

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        z0, y0, x0 = self.indices[idx]

        z1 = z0 + self.window_size
        y1 = y0 + self.crop_size
        x1 = x0 + self.crop_size

        sub_volume = self.volume[
            z0:z1,
            y0:y1,
            x0:x1
        ]

        sub_volume = torch.from_numpy(
            sub_volume
        ).float()

        sub_volume = sub_volume.unsqueeze(0)

        return sub_volume


############################################################
# Inference Dataset
############################################################

class VEMInferenceDataset(Dataset):

    """
    Output:

    [1,F,H,W]
    """

    def __init__(self, volume_dir):

        super().__init__()

        self.volume = load_volume_from_folder(
            volume_dir
        )

    def __len__(self):

        return 1

    def __getitem__(self, idx):

        volume = torch.from_numpy(
            self.volume
        ).float()

        volume = volume.unsqueeze(0)

        return volume


############################################################
# Test
############################################################

if __name__ == "__main__":

    volume_dir = "/path/to/vem"

    ########################################################
    # train dataset
    ########################################################

    train_dataset = VEMTrainDataset(
        volume_dir=volume_dir,
        window_size=16,
        crop_size=32,
        stride_z=8,
        stride_xy=16
    )

    print(
        "train dataset length:",
        len(train_dataset)
    )

    sample = train_dataset[0]

    print(
        "train sample shape:",
        sample.shape
    )

    ########################################################
    # inference dataset
    ########################################################

    infer_dataset = VEMInferenceDataset(
        volume_dir
    )

    volume = infer_dataset[0]

    print(
        "inference volume shape:",
        volume.shape
    )
