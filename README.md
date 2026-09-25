# ReStack-Mamba

Code for serial-section electron microscopy registration. Start with an ordered image stack, generate simulated elastic distortions if needed, then run **LEA alignment → two-stage StackMamba refinement → registered volume**. The original stack remains the reference for evaluating the simulated data.

## 1. Environment

Use Python 3.10, CUDA 12.1, and PyTorch 2.5.1. Install a CUDA-compatible PyTorch build first, then:

```bash
pip install -r requirements.txt
pip install -e .
```

## 2. Generate simulated data

Prepare a folder of grayscale PNG slices named `0.png`, `1.png`, … in section order. All slices must have the same height and width. After installing the environment above, run:

```bash
python tools/generate_elastic_simulation.py \
  --input_xy <original_slices> \
  --output_xy <simulated_slices> \
  --inverse_flow <simulation_dir>/displacement.npy \
  --alpha 3 --sigma 0.08 --seed 42
```

| Parameter | What to pass |
| --- | --- |
| `--input_xy` | Folder containing the original, undistorted PNG slices. |
| `--output_xy` | Folder where the simulated PNG slices will be saved. |
| `--inverse_flow` | Output NPY displacement file, also used by the strength-scaling command below. |
| `--alpha` | Displacement multiplier; larger values produce stronger distortion. |
| `--sigma` | Spatial smoothing scale as a fraction of image width. |
| `--seed` | Random seed; reuse it to repeat the same simulation. |

The script applies a smooth 2D elastic deformation to each section and keeps the first section unchanged. It saves the distorted PNGs, a displacement array of shape `(Z, 2, H, W)`, and `simulation_metadata.json` in the parent of `--output_xy`. Use a separate output folder for each simulation. Pass the simulated PNG folder as `<training_slices>` or `<input_slices>` in the commands below.

To compare several distortion strengths using the **same deformation pattern**, reuse the saved displacement file:

```bash
python tools/generate_scaled_simulations_from_flow.py \
  --input_xy <original_slices> \
  --reference_inverse_flow <simulation_dir>/displacement.npy \
  --reference_alpha 3 \
  --alphas 1 2 3 4 5 \
  --output_root <severity_dir>
```

`--reference_inverse_flow` is the file saved by the first command; `--reference_alpha` must match the `--alpha` used to create it. `--alphas` lists the desired strengths. Results are saved under `<severity_dir>/alpha_<value>/simulation/XY/`, with settings recorded in each `simulation_metadata.json`.

## 3. Training

**Step 1: train LEA.**

```bash
python -m mamba_reg.train_lea_local_aligner \
  --volume_dir <training_slices> --save_dir <lea_dir> --gpu 0
```

`training_slices` is a folder of ordered section images. `lea_dir` is where training logs and `best.pth` are saved. `--gpu` selects the CUDA device.

**Step 2: train StackMamba.** This part has two passes. First, run LEA once to produce the input volume:

```bash
python -m mamba_reg.inference_lea_stack \
  --input_dir <training_slices> --model_path <lea_dir>/best.pth \
  --output_dir <lea_output> --gpu 0
```

**Mamba Stage 1:** train on the LEA volume, then run the trained Stage 1 model once to create the Stage 2 training input.

```bash
python -m mamba_reg.train_stack_residual_mamba \
  --volume <lea_output>/registered_volume_uint8.npy \
  --save_dir <stage1_dir> --gpu 0 \
  --feature_dim 48 --mamba_depth 4 --residual_max_flow 1.0

python -m mamba_reg.inference_stack_residual_mamba \
  --volume <lea_output>/registered_volume_uint8.npy \
  --checkpoint <stage1_dir>/best.pth --output_dir <stage1_output> --gpu 0
```

**Mamba Stage 2:** train on the Stage 1 output.

```bash
python -m mamba_reg.train_stack_residual_mamba \
  --volume <stage1_output>/registered_volume_uint8.npy \
  --save_dir <stage2_dir> --gpu 0 \
  --feature_dim 48 --mamba_depth 4 --residual_max_flow 1.0
```

`--volume` takes a `Z,Y,X` NPY volume. For paired training, also pass `--target_volume <reference_volume.npy>` to both StackMamba training commands. `--save_dir` contains `best.pth`; `--checkpoint` loads that file for the intermediate inference; `--output_dir` stores the registered volume. Other training options are listed by each command's `--help`.

## 4. Inference

```bash
python scripts/reproduce_inference.py \
  --input-xy <input_slices> \
  --lea-checkpoint <lea_dir>/best.pth \
  --stage1-checkpoint <stage1_dir>/best.pth \
  --stage2-checkpoint <stage2_dir>/best.pth \
  --output <output_dir> --gpu 0
```

`input_slices` contains consecutively numbered PNG files (`0.png`, `1.png`, …). The three checkpoint arguments load the trained LEA, Stage 1, and Stage 2 weights. `output_dir` receives `final_volume.npy` and PNG views in `views/XY`, `views/XZ`, and `views/YZ`. If the stack is already aligned, replace `--lea-checkpoint` with `--aligned-volume <aligned_volume.npy>`.
