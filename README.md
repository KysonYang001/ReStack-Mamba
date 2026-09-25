# ReStack-Mamba

Code for two-stage residual registration of serial-section volume electron microscopy stacks. This source release was assembled from the **15 server's method code** and validated against its four selected final checkpoints. The checkpoint files will be published separately; their SHA-256 checksums and metadata are included here. The six simulated datasets use the same frozen Stage 1/Stage 2 pair; the real Kasthuri11 experiment uses a separately trained pair. See [checkpoint_manifest.json](checkpoint_manifest.json) for the dataset-to-weight mapping.

## What is in this repository

| Path | Purpose |
| --- | --- |
| `mamba_reg/` | Exact model, training, local-aligner, inference, and metric source snapshot from the 15 server |
| `scripts/prepare_simulated_data.py` | Download and simulate the six evaluated OpenOrganelle stacks |
| `scripts/train_two_stage.py` | Train Stage 1, infer its fixed outputs, then train Stage 2 |
| `scripts/reproduce_inference.py` | Local alignment, two Mamba stages, two axial trimmed-mean passes, and XY evaluation |
| `tools/`, `paper_comparison/` | Data conversion, simulation, export, and evaluation used by the pipeline |
| `analysis_tools/` | Local post-analysis code for volume-wide NCC and MI |
| `checkpoints/` | Four selected `best.pth` checkpoints; files pending separate publication |
| `checkpoint_metadata.json`, `SHA256SUMS`, `SOURCE_SHA256SUMS` | Checkpoint-recorded settings and binary/source checksums |
| `provenance/` | Original 15-server run scripts; these contain server-specific absolute paths and are provided for audit |

The final simulated-volume model takes the [official vEMRec](https://github.com/zhangzhenbang2021/vEMRec) OpenOrganelle elastic-registration output as its local-alignment input. The vEMRec code and `openog.pth` are third-party artifacts and are **not redistributed here**. Obtain them from the upstream repository's pre-trained-model link and place the checkpoint at `src/elastic/premodel/openog.pth`. The model used on the 15 server has SHA-256 `c02a92f0070de16422cab682e633b957e670a880880f787c308324468447dcc1`.

## Environment

Use Python 3.10+ and a CUDA-compatible PyTorch installation. The 15-server environment used PyTorch `2.5.1+cu121`, CUDA `12.1`, `mamba-ssm==2.2.4`, `causal-conv1d==1.6.2.post1`, NumPy `2.2.6`, and SciPy `1.15.3`. After installing matching PyTorch and CUDA packages:

```bash
pip install -r requirements.txt
pip install -e .
sha256sum -c SOURCE_SHA256SUMS
```

Install vEMRec in its own environment following its upstream instructions. If local alignment has already been generated, use `--aligned-volume` and a vEMRec installation is unnecessary for the two StackMamba stages.

## Reproduce inference

OpenOrganelle data are downloaded directly from the public Janelia S3 N5 store by the included utility. It transposes and bilinearly resizes scale `s4` sections to `800×800`. The simulator uses seed `42`, Gaussian scale `0.08`, and deformation magnitude `4` for `jrc_mus_liver` or `3` for the other five datasets. The source server's `orignal` directory spelling is retained for path compatibility.

```bash
python scripts/prepare_simulated_data.py --data-root data

python scripts/reproduce_inference.py \
  --input-xy data/jrc_mus_kidney/simulated/XY \
  --reference-xy data/jrc_mus_kidney/orignal/XY \
  --vemrec-root /path/to/vEMRec \
  --vemrec-python /path/to/vemrec-env/bin/python \
  --checkpoint-set shared --gpu 0 --output runs/jrc_mus_kidney
```

The result is `runs/jrc_mus_kidney/final_volume.npy`; per-section XY metrics are in `XY_metrics.csv`, all-view PSNR/SSIM in `metrics_volume/`, volume-wide NCC/MI in `ncc_mi_3d.csv`, and image slices in `views/XY`, `views/XZ`, and `views/YZ`. The script resumes existing stage outputs and checks section continuity, shape, checkpoint hashes, and volume type. Replace the dataset name for the other five simulated volumes. Use `--checkpoint-set kasthuri11` for the 1,024-section Kasthuri11 central crop. Real-data results are qualitative because no undeformed reference is available.

You can supply an existing local-alignment volume instead:

```bash
python scripts/reproduce_inference.py \
  --input-xy /path/to/numeric/XY \
  --aligned-volume /path/to/local_alignment_uint8.npy \
  --checkpoint-set shared --output runs/example
```

## Retrain

The shared simulated-volume stages were trained on **paired** locally aligned/reference volumes from `jrc_mus_kidney`, `jrc_mus_heart`, `jrc_mus_pancreas`, and `jrc_mus_skin`. `jrc_mus_liver` and `jrc_mus_liver3` were evaluated with those frozen checkpoints. Kasthuri11 stages were trained without reference targets. `checkpoint_metadata.json` records the actual hyperparameters embedded in each selected checkpoint: the released StackMamba stages use width `48`, depth `4`, and maximum residual flow `1.0`. The shared checkpoints were selected at epoch `5` for both stages; Kasthuri11 at epochs `2` and `10`. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact preparation and training commands.

The training implementation does not force all random seeds, so retraining is not guaranteed to yield byte-identical weights. Inference with the selected weights requires the separately published checkpoints, the specified inputs, and the settings recorded here.

## Data and attribution

This repository does not contain the source microscopy volumes or the vEMRec model. The six simulated volumes are derived from the [OpenOrganelle datasets](https://openorganelle.janelia.org/); obtain Kasthuri11 data from its original source and prepare the same 1,024-section central crop for the real-data comparison. Cite the dataset sources and [vEMRec](https://github.com/zhangzhenbang2021/vEMRec) when using their data or local-alignment implementation.
