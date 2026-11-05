# 🔁 Reproducing the paper

This page lists the commands behind the FoSA rows of the paper's tables. Each command runs inside the
conda environment of its foundation model (`bash scripts/setup_env.sh <model>`, then
`conda activate fosa-<model>`; see the [README](../README.md#-setup)).

## Contents

1. [Datasets](#1-datasets)
2. [Protocol](#2-protocol)
3. [Commands per table](#3-commands-per-table)
4. [Outputs](#4-outputs)
5. [Backbone cache](#5-backbone-cache)
6. [Determinism](#6-determinism)

## 1. Datasets

`scripts/reproduce.sh` expects the five benchmarks under one folder, `$DATA_ROOT`. `main.py` takes
any one of them with `--dataset <name> --dataset-path <folder>`.

| `--dataset` | Folder under `$DATA_ROOT` | Layout | Clips (T = 3) |
| :--- | :--- | :--- | :---: |
| `nyu_video` | `NYUv2` | `<scene>/<frame>.h5`, each with `rgb` [3,H,W] uint8 and `depth` [H,W] in meters. Anchors: 500 pixels of the dense depth per frame, uniformly at random | 543 (8 scenes) |
| `void` | `VOID/void_500` (`VOID/void_150` for Table VI) | A density folder of the VOID release: `test_image.txt`, `test_sparse_depth.txt`, `test_ground_truth.txt`, whose entries (`void_500/data/<scene>/image/<t>.png`, ...) are relative to its parent folder | 263 |
| `kitti_video` | `KITTI-DC` | `val/<drive>/{rgb,depth,lidar}/<frame>.png` (uint16 / 256 = meters) and `val/<drive>/intrinsics/<frame>.txt` (3×3 K, needed only for `--num-lidar-lines` < 64). Resized to 375×1242 | 1137 (13 drives) |
| `dydtof` | `DyDToF` | `<shot>/ColorImage/<frame>.jpeg`, `<shot>/DepthMap/<frame>.npy` (meters). Shots: ares_shot1, ba_shot1, cabin_shot1, church_shot1, coffe_shot1, museum_shot1, res_shot1, royale_shot1. Anchors: 500 random pixels of the dense depth per frame | 1506 |
| `ddad_vid` | `DDAD` | `rgb/`, `gt/`, `hints/` (`<frame>.png`, uint16 / 256 = meters): the 50 validation scenes back to back, in the order of `utils/dataloader/ddad_valset_info.json` | 1293 |

The datasets are not redistributed here; obtain them from their original sources (listed in
[RESULTS.md](RESULTS.md#-benchmarks-and-protocol)) under their own licenses.

## 2. Protocol

- Every sequence is cut into non-overlapping clips of 3 frames that never cross a sequence boundary.
  R-TGM carries over from each clip to the next one of the same sequence.
- Random anchors (NYUv2, DyDToF) are drawn with a generator seeded by the clip index alone
  (`--random-seed`, default 2025), so every run sees the same anchors.
- VDA, DVD, and DAv3 see the frames center-cropped to multiples of 14 (`input_patch_size: 14`);
  UniDepth V2 sees full frames.
- FoSA fits in depth for UniDepth V2 and in inverse depth for the other backbones (`domain` in
  `config/<model>.yaml`). All other settings are shared; see
  [METHOD.md](METHOD.md#11-defaults-in-this-repository).

## 3. Commands per table

```bash
export DATA_ROOT=/path/to/datasets     # laid out as in Sec. 1
export CACHE_ROOT=/path/to/cache       # optional: keep the backbone outputs for later runs

# Table I (UniDepth V2, FoSA row) and the UniDepth V2 rows of Table V: the five benchmarks
bash scripts/reproduce.sh unidepth main

# Tables II-IV (VDA, DVD, DAv3) and their rows of Table V, one environment each
bash scripts/reproduce.sh vda
bash scripts/reproduce.sh dvd
bash scripts/reproduce.sh da3

# Table VI (sparser measurements, UniDepth V2)
bash scripts/reproduce.sh unidepth sparse
```

The temporal-consistency tables (TEPE on NYUv2, KITTI-DC, and DyDToF) use the TEPE column of the
same runs. Table VI changes only the measurements:

| Benchmark | Setting | `main.py` option |
| :--- | :--- | :--- |
| KITTI-DC | 8 of the 64 LiDAR lines | `--num-lidar-lines 8` |
| DyDToF | 50 random anchors per frame | `--num-sparse-samples 50` |
| NYUv2 | 50 random anchors per frame | `--num-sparse-samples 50` |
| VOID | the 150-point density | `--dataset-path $DATA_ROOT/VOID/void_150` |
| DDAD | each hint kept with probability 0.1 | `--sparse-keep-ratio 0.1` |

A single run, for example UniDepth V2 on KITTI-DC with timing:

```bash
python main.py --config config/unidepth.yaml --dataset kitti_video \
    --dataset-path $DATA_ROOT/KITTI-DC --profile
```

`--profile` adds the backbone time and the FoSA time per clip and the GPU memory that FoSA allocates
on top of the backbone (peak per clip and mean per frame).

## 4. Outputs

Each run prints RMSE, MAE, and TEPE in meters and writes `outputs/<time>_<name>/`:

- `eval_results.txt`: one line per clip (RMSE, MAE, TEPE, and the Fourier order FoSA chose); the
  last line holds the means over clips.
- `args.json`: the exact command-line arguments.
- `vis/<clip>_<frame>.png` with `--save-vis --vis-every N`: RGB, anchors, and backbone depth on top;
  ground truth, FoSA depth, and |FoSA − GT| below.

## 5. Backbone cache

`--base-cache <folder>` (set by `scripts/reproduce.sh` when `CACHE_ROOT` is given) stores the
backbone output of every clip (fp32, one file per clip) and reuses it in later runs, so FoSA
settings can be compared without running the backbone again. A `meta.json` in the folder refuses a
cache built for another backbone, dataset, clip length, or crop.

## 6. Determinism

- FoSA and the data pipeline are deterministic: given the same backbone output, every metric is
  identical across runs.
- UniDepth V2 (commit `bebc4b2`) and DAv3 reproduce the backbone outputs of the paper bit for bit
  in the environments the paper used.
- VDA matches to about 3e-5 relative; the same difference appears between two runs of the same code,
  so it comes from GPU kernel selection.
- DVD's diffusion pipeline is not deterministic across runs, and its numbers move slightly with it.
