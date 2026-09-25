# ReStack-Mamba

PyTorch implementation of serial-section electron microscopy registration. The pipeline first aligns adjacent sections with LEA, then refines the volume with two StackMamba stages.

## Install

Install a CUDA-compatible PyTorch build and then run `pip install -r requirements.txt && pip install -e .`.

## 1. LEA: train and infer

```bash
python -m mamba_reg.train_lea_local_aligner \
  --volume_dir <training_slices> --save_dir <lea_weights> --gpu 0

python -m mamba_reg.inference_lea_stack \
  --input_dir <input_slices> --model_path <lea_weights>/best.pth \
  --output_dir <lea_output> --gpu 0
```

`--volume_dir` and `--input_dir` are folders of consecutively ordered image slices. `--save_dir` stores LEA checkpoints; `--model_path` selects one checkpoint; `--output_dir` stores the registered PNG slices and `registered_volume_uint8.npy`. `--gpu` selects the CUDA device. Optional `--coarse_align` enables rigid edge-based pre-alignment; the other LEA options are shown by `--help`.

## 2. StackMamba: train and infer

Train Stage 1 on the LEA output, run Stage 1 inference, then train Stage 2 on that result. For paired simulated data, add `--target_volume <reference_volume.npy>` to each training command; omit it for unpaired data.

```bash
python -m mamba_reg.train_stack_residual_mamba \
  --volume <lea_output>/registered_volume_uint8.npy \
  --save_dir <stage1_weights> --gpu 0

python -m mamba_reg.inference_stack_residual_mamba \
  --volume <lea_output>/registered_volume_uint8.npy \
  --checkpoint <stage1_weights>/best.pth \
  --output_dir <stage1_output> --gpu 0

python -m mamba_reg.train_stack_residual_mamba \
  --volume <stage1_output>/registered_volume_uint8.npy \
  --save_dir <stage2_weights> --gpu 0

python -m mamba_reg.inference_stack_residual_mamba \
  --volume <stage1_output>/registered_volume_uint8.npy \
  --checkpoint <stage2_weights>/best.pth \
  --output_dir <stage2_output> --gpu 0
```

`--volume` and optional `--target_volume` take `Z,Y,X` NPY volumes. `--save_dir` stores training checkpoints; `--checkpoint` loads a trained weight; `--output_dir` stores the registered volume; `--gpu` selects the CUDA device. The Stage 2 volume is `<stage2_output>/registered_volume_uint8.npy`. Run `--help` on either module for model, crop, and optimizer options.

## 3. Final volume and outputs

The final axial aggregation is applied twice:

```bash
python tools/trimmed_mean_volume_z.py --input <stage2_output>/registered_volume_uint8.npy --output <output>/trim_pass1.npy --radius 2 --trim 1
python tools/trimmed_mean_volume_z.py --input <output>/trim_pass1.npy --output <output>/final_volume.npy --radius 2 --trim 1
```

`--input` is the source volume, `--output` names the saved NPY file, `--radius` sets the axial neighborhood, and `--trim` removes extreme values before averaging. The final result is `<output>/final_volume.npy`. Use `paper_comparison/export_volume_xy.py` and `tools/make_orthogonal_slices.py` to export XY, XZ, and YZ PNG views.

Selected trained weights are being published separately. `checkpoint_manifest.json` lists their expected paths and hashes.
