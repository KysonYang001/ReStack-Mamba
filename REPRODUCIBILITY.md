# Reproduction details

This document distinguishes the exact final inference path from alternative experiments in the server archive. `provenance/` contains the original scripts with absolute server paths; the portable scripts under `scripts/` use the same model calls and settings.

## Dataset map

| Dataset | OpenOrganelle ID | Sections | Simulated alpha | Checkpoint pair |
| --- | --- | ---: | ---: | --- |
| `jrc_mus_liver` | `jrc_mus-liver` | 558 | 4 | `shared` |
| `jrc_mus_kidney` | `jrc_mus-kidney` | 1,387 | 3 | `shared` |
| `jrc_mus_heart` | `jrc_mus-heart-1` | 1,061 | 3 | `shared` |
| `jrc_mus_pancreas` | `jrc_mus-pancreas-4` | 898 | 3 | `shared` |
| `jrc_mus_skin` | `jrc_mus-skin-1` | 1,231 | 3 | `shared` |
| `jrc_mus_liver3` | `jrc_mus-liver-3` | 1,127 | 3 | `shared` |
| `kasthuri11` | Original real-data source | 1,024 | — | `kasthuri11` |

The OpenOrganelle volumes were downloaded at scale `s4`, transposed, and resized bilinearly to `800×800`. Every section except the first receives an independently sampled 2D smooth deformation. The simulator's seed is `42` and Gaussian scale is `0.08`.

## Exact released inference sequence

1. Run upstream vEMRec elastic local alignment with `openog.pth`, `--iters 3 --iter_T 2 --sigma 3.0 --r 1 --L 1`.
2. Run `mamba_reg.inference_stack_residual_mamba` with the shared or Kasthuri11 Stage 1 checkpoint: `--z_size 16 --z_stride 8 --crop_size 256 --xy_stride 192 --feature_dim 48 --mamba_depth 4 --residual_max_flow 1.0 --flow_scale 1.0 --fusion_mode image --image_z_residual_smooth_sigma 0.35`.
3. Feed the fixed Stage 1 output to the independent Stage 2 checkpoint with identical inference settings.
4. Apply `trimmed_mean_volume_z.py --radius 2 --trim 1` twice.
5. Export the final XY/XZ/YZ views and evaluate against the undeformed reference for simulated data. The volume-wide NCC/MI helper is from the local paper-analysis workspace; the model, training, and inference source is from the 15 server.

The portable `scripts/reproduce_inference.py` executes this sequence and verifies checkpoint SHA-256 hashes before inference.

## Retrain the shared model

First prepare the four paired simulated datasets and generate their local-alignment volumes. For example:

```bash
python scripts/prepare_simulated_data.py --data-root data \
  --datasets jrc_mus_kidney jrc_mus_heart jrc_mus_pancreas jrc_mus_skin

python scripts/reproduce_inference.py \
  --input-xy data/jrc_mus_kidney/simulated/XY \
  --vemrec-root /path/to/vEMRec \
  --vemrec-python /path/to/vemrec-env/bin/python \
  --checkpoint-set shared --output runs/alignment_kidney --local-align-only
```

Repeat the local-alignment command for heart, pancreas, and skin. Then train both stages. The script reads the actual training settings saved in the released checkpoints and invokes the 15-server training implementation. It trains Stage 1 on four aligned/reference pairs, infers Stage 1 volumes with frozen weights, and trains Stage 2 on those fixed outputs:

```bash
python scripts/train_two_stage.py --checkpoint-set shared --output runs/retrained_shared \
  --dataset jrc_mus_kidney runs/alignment_kidney/local_alignment/registered_volume_uint8.npy data/jrc_mus_kidney/orignal/volume_uint8.npy \
  --dataset jrc_mus_heart runs/alignment_heart/local_alignment/registered_volume_uint8.npy data/jrc_mus_heart/orignal/volume_uint8.npy \
  --dataset jrc_mus_pancreas runs/alignment_pancreas/local_alignment/registered_volume_uint8.npy data/jrc_mus_pancreas/orignal/volume_uint8.npy \
  --dataset jrc_mus_skin runs/alignment_skin/local_alignment/registered_volume_uint8.npy data/jrc_mus_skin/orignal/volume_uint8.npy
```

For Kasthuri11, supply its locally aligned real-data volume without a target:

```bash
python scripts/train_two_stage.py --checkpoint-set kasthuri11 \
  --dataset kasthuri11 /path/to/kasthuri11_aligned_uint8.npy - \
  --output runs/retrained_kasthuri11
```

`--dry-run` prints training commands without launching GPU computation. The shared checkpoint loss uses paired reference terms; the Kasthuri11 checkpoint uses the unpaired terms shown in `checkpoint_metadata.json`. The checkpoint contains the model, optimizer, scaler, selected epoch, and saved argument values.

## Limits of exact replication

The pretrained local-alignment model and original microscopy data are upstream dependencies. The original training process did not pin all RNG seeds. GPU, CUDA, and `mamba-ssm` builds can also affect retraining. Use the included frozen checkpoints and their checksums to reproduce released inference outputs before attempting retraining.
