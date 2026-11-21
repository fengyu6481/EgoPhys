# EgoPhys: Estimating Peak Contact Force and Mechanical Work from Egocentric Manipulation Video

EgoPhys predicts peak contact force in newtons and contact-phase mechanical work in joules from a head-mounted RGB clip. It uses frozen DINOv3 and Depth Anything 3 encoders.

Contact-Aware Spatial Aggregation (CASA) projects the two 7×7 feature grids to 192 channels, exchanges information with bidirectional cross-attention, and applies spatial self-attention and learned weighted pooling. Target-Specific Multi-Expert Temporal Routing (TMTR) applies separate six-block dilated temporal convolutions to semantic, event, and motion features. Independent gates and masked attention pool the expert outputs for the two non-negative predictions.

## Install

Use Python 3.10 or later and a PyTorch build with CUDA for training. The reference Linux environment used Python 3.11, PyTorch 2.7.1, NumPy 2.2.6, and safetensors 0.5.3.

```bash
python -m pip install -e .
# Add the dependencies for data preparation and feature extraction.
python -m pip install -e '.[prepare]'
```

## Data

Download and unpack the gripper recordings listed in `data/boundaries.json`. Follow the upstream tools for the Aria exports, gripper pose, calibration, and force/torque files. The included boundaries are the reviewed start and end times for the 336 clips. The fixed split includes 264 training, 36 validation, and 36 test clips from 22 recordings across 18 scenes. It is a clip-level split with one open and one close test clip per scene. [Hoi! dataset](https://hoi-dataset.ethz.ch/) · [Hoi! dataset tools](https://github.com/timengelbracht/hoi-dataset-tools)

### Generate labels and features

Set `HOI_ROOT` to the unpacked dataset. For extraction, set `DINO_WEIGHTS` and `DA3_WEIGHTS` to the local model directories for DINOv3 ViT-L/16 and DA3-LARGE-1.1. Install the upstream DA3 package using its installation instructions. The intermediate feature extractor uses the DA3 backbone interface from the 3fe327a checkout. [DINOv3 ViT-L/16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) · [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) · [DA3-LARGE-1.1](https://huggingface.co/depth-anything/DA3-LARGE-1.1)

```bash
egophys-manifest --boundaries data/boundaries.json --output prepared/manifests \
  --labels-dir prepared/labels --geometry-dir prepared/geometry

for window in prepared/manifests/windows/*.json; do
  uid="$(basename "$window" .json)"
  scene="${uid%%__*}"
  recording="${uid#*__}"
  egophys-labels --dataset-root "$HOI_ROOT" --confirmed-json "$window" \
    --output-root "prepared/labels/$uid" --scene "$scene" --recording "$recording" || exit
  egophys-geometry --dataset-root "$HOI_ROOT" --t2-label-root "prepared/labels/$uid" \
    --output-root "prepared/geometry/$uid" --scene "$scene" --recording "$recording" || exit
  egophys-extract --dataset-root "$HOI_ROOT" --label-root "prepared/labels/$uid" \
    --recording-uid "$uid" --dino "$DINO_WEIGHTS" --da3 "$DA3_WEIGHTS" \
    --output prepared/features || exit
done
```

Peak force is the maximum force-vector norm over the clip. Work is the absolute value of the summed signed force–velocity integrals across the runs where force exceeds 2 N. The label and geometry scripts produce sequence targets and align the auxiliary mean-force and loaded-envelope targets to the RGB frames. Encoder feature extraction uses RGB frames and their timestamps.

### Use existing caches

`--manifest` points to a JSON object with a `recordings` array. Each entry has `recording_uid`, `label_root`, and `geometry_root`. Paths can be absolute or relative to the manifest. Directory names in the split have the form `recording_uid__sample_id`.

| Cache | File under its root | Required contents |
| --- | --- | --- |
| Compact | `<recording_uid>/<sample_id>.npz` | Clip identity, nanosecond frame timestamps, RGB-valid mask, class token in slot 0 of `tokens[T,5,1024]`, and the token names |
| Spatial | `<recording_uid>/<sample_id>.npz` | Clip identity, timestamps, RGB-valid mask, `dino_grid_7x7[T,7,7,1024]`, and `dino_patch_mean[T,1024]` |
| DA3 | `samples/<recording_uid>__<sample_id>.safetensors` | `sample_uid` metadata, timestamps, DA3-valid mask under `da3_token_valid_mask`, and `da3_intermediate_grid_7x7[T,7,7,2048]` |

Repeat `--da3-dir` for several DA3 cache roots. The feature files and annotation manifests must identify the same clip and agree on the frame timestamps.

## Train and evaluate

```bash
features=(--compact-dir prepared/features/compact --spatial-dir prepared/features/spatial \
          --da3-dir prepared/features/da3)
labels=(--manifest prepared/manifests/recordings.json --split data/split.json)

egophys preflight "${labels[@]}" "${features[@]}" --rgb-root "$HOI_ROOT"

egophys train "${labels[@]}" "${features[@]}" --variant full --seed 1 \
  --epochs 60 --batch-size 2 --output runs/full_seed1

egophys evaluate "${labels[@]}" "${features[@]}" --checkpoint runs/full_seed1/best.pt \
  --partition test --output runs/full_seed1_test

egophys predict --split data/split.json "${features[@]}" \
  --checkpoint runs/full_seed1/best.pt --output runs/full_seed1_predictions
```

Train with seeds 1 through 6 for the six-seed protocol. Training selects the checkpoint by the sum of validation MAEs divided by the two train-only scales. It saves the configuration, per-epoch history, and validation metrics. Evaluation reports MAE, MedianAE, RMSE, and MRE independently for both targets. Use `--precision bf16` on compatible GPUs and the same precision for training and evaluation. Use `--cache-memory` during training when approximately 32 GB of host memory is available for decoded features.

Prediction accepts a split file or `--ids` with a JSON array of recording-qualified clip IDs. It writes `predictions.json` and `submission.zip`. Each prediction is keyed by its sequence directory name and has two fields, `max_force_n` and `work_j`. The export checks that these directory names are unique.

### Experiment configurations

| Table | Configuration | `--variant` |
| --- | --- | --- |
| 1 | DINOv3 only, DA3 only, both features | `dino`, `da3`, `dino_da3` |
| 2 | #A, #B, #C, full model | `dino_da3`, `casa`, `tmtr`, `full` |
| 3 | Semantic, event, motion, all (concat), all (gate) | `semantic`, `event`, `motion`, `concat`, `full` |

The component ablation uses the same normalized scalar loss throughout and enables the two frame-level auxiliary losses for the full model. Expert comparisons share the full training objective.

### Paper-reported test results

Mean ± standard deviation over six seeds in the EgoPhys paper.

| Peak contact force | MAE (N) | MedianAE (N) | RMSE (N) | MRE |
| --- | ---: | ---: | ---: | ---: |
| DINOv3 only | 9.320 ± 0.189 | 5.696 ± 0.337 | 13.827 ± 0.290 | 0.584 ± 0.015 |
| DA3 only | 9.318 ± 0.770 | 5.702 ± 0.523 | 13.689 ± 1.334 | 0.596 ± 0.039 |
| DINOv3 + DA3 | 8.796 ± 1.230 | 5.702 ± 0.801 | 12.689 ± 1.949 | 0.600 ± 0.068 |
| EgoPhys | **5.205 ± 0.584** | **3.492 ± 0.594** | **7.132 ± 0.765** | **0.301 ± 0.031** |

| Mechanical work | MAE (J) | MedianAE (J) | RMSE (J) | MRE |
| --- | ---: | ---: | ---: | ---: |
| DINOv3 only | 1.700 ± 0.032 | 0.665 ± 0.139 | 3.088 ± 0.094 | 1.948 ± 0.347 |
| DA3 only | 1.656 ± 0.128 | 0.823 ± 0.037 | 2.751 ± 0.234 | 2.146 ± 0.265 |
| DINOv3 + DA3 | 1.624 ± 0.124 | 0.852 ± 0.050 | 2.695 ± 0.226 | 2.034 ± 0.337 |
| EgoPhys | **0.894 ± 0.081** | **0.427 ± 0.045** | **1.508 ± 0.203** | **1.237 ± 0.189** |

## License

EgoPhys code is released under the MIT License. The Hoi! split and reviewed boundary metadata identify clips from Hoi! © 2025–2026 ETH Zurich and collaborators, distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The upstream encoders and their weights follow their respective licenses.
