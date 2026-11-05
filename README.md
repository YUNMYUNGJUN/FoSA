<div align="center">

# 〰️ FoSA: Fast Video Depth Completion via Fourier Scale-and-Shift Alignment

📰 **IEEE TCSVT** (under review)

Myungjun Yun · Jin-Hwi Park

[CSILab](https://www.csilab.cau.ac.kr/home) · [GSAIM](https://gsaim.cau.ac.kr), <img src="assets/cau-logo.png" alt="Chung-Ang University" height="20" align="absmiddle"> [Chung-Ang University](https://www.cau.ac.kr)

📄 Paper (coming soon) · [🧮 Method](docs/methodology.md) · [🚀 Run](docs/running.md)

<img src="assets/fosa-overview.png" alt="FoSA overview: Stage 1 expands scale and shift into damped Fourier fields; Stage 2 (R-TGM) regularizes the correction residual over time; each frame is aligned by a closed-form linear solve" width="100%">

</div>

## Abstract

Depth foundation models predict relative structure reliably but fail to recover metric depth,
as monocular reconstruction is ill-posed up to scale. The standard remedy fits a global affine
transform against the sparse measurements, but with only two degrees of freedom it applies a
single uniform correction to the entire image and is therefore limited in expressiveness.
Test-time adaptation methods recover metric depth effectively by adapting predictions to
individual inputs, yet require costly iterative backpropagation, limiting their practical
deployment. To address these limitations, we propose **Fo**urier **S**cale-and-Shift
**A**lignment (**FoSA**), a lightweight, training-free framework that represents scale and shift
as spatially varying fields expanded in a Fourier basis, avoiding the need for iterative
optimization. To automatically control model complexity, each frequency component is attenuated
according to the local spacing of sparse measurements, so that the effective degrees of freedom
follow the Nyquist limit of the observations without manual tuning. Furthermore, we introduce
**R**esidual **T**emporal **G**radient **M**atching (**R-TGM**), which penalizes changes in the
correction residual rather than in the coefficients, preserving the temporal behavior of the base
model without requiring optical flow or camera pose. This enables robust, closed-form alignment
under varying sensor densities, diverse backbone types, and dynamic scenes across multiple indoor
and outdoor benchmarks.

## Benchmarks and backbones

FoSA is training-free: it aligns the output of a frozen depth backbone to sparse metric
measurements, so there is nothing to train and no dataset of our own to download.

| Benchmark | Sparse pattern (default → sparser) | Paper |
| :--- | :---: | :--- |
| NYUv2 | Random 500 → 50 points | [Indoor Segmentation and Support Inference from RGBD Images](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) (ECCV 2012) |
| VOID | 500 → 150 points | [Unsupervised Depth Completion from Visual Inertial Odometry](https://github.com/alexklwong/void-dataset) (RA-L 2020) |
| DyDToF | Random 500 → 50 points | [Consistent Direct Time-of-Flight Video Depth Super-Resolution](https://github.com/facebookresearch/DVSR) (CVPR 2023) |
| KITTI-DC | 64-line → 8-line LiDAR | [Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) (CVPR 2012) |
| DDAD | Random 20% → 2% of LiDAR points | [3D Packing for Self-Supervised Monocular Depth Estimation](https://github.com/TRI-ML/DDAD) (CVPR 2020) |

| Backbone | Input | FoSA fits the fields in | Paper |
| :--- | :---: | :---: | :--- |
| UniDepth V2 | Single image | Depth | [UniDepthV2](https://arxiv.org/abs/2502.20110) ([code](https://github.com/lpiccinelli-eth/UniDepth)) |
| VDA | Video | Inverse depth | [Video Depth Anything](https://arxiv.org/abs/2501.12375) ([code](https://github.com/DepthAnything/Video-Depth-Anything)) |
| DVD | Video | Inverse depth | [DVD](https://arxiv.org/abs/2603.12250) ([code](https://github.com/EnVision-Research/DVD)) |
| DAv3 | Single image | Inverse depth | [Depth Anything 3](https://arxiv.org/abs/2511.10647) ([code](https://github.com/ByteDance-Seed/Depth-Anything-3)) |

## Results

**Setup.** All sparse depth alignment methods (LS Solver, Polynomial, Piecewise, Ours) are applied
to the same backbone predictions and the same sparse measurements. RMSE and MAE pool the valid
pixels of all frames, and TEPE measures the frame-to-frame change of the aligned depth after
warping with GMFlow optical flow. All metrics are in meters and lower is better. The ground truth
of KITTI-DC and DDAD is clipped at 80 m. **Bold** marks the best value in each column (or row
group), following the paper.

### Comparison with depth completion methods

UniDepth V2 backbone for every sparse depth alignment method. **TTA**: test-time adaptation of the
backbone. **Align**: sparse depth alignment of a frozen backbone's output.

| Method | Type | NYUv2 RMSE | NYUv2 MAE | KITTI-DC RMSE | KITTI-DC MAE | DDAD RMSE | DDAD MAE | DyDToF RMSE | DyDToF MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Zero-DC | TTA | 0.152 | 0.097 | 1.515 | 0.509 | 7.163 | 3.621 | 3.399 | 2.038 |
| Marigold-DC | TTA | 0.135 | **0.053** | 1.483 | 0.550 | 6.401 | 2.708 | 2.662 | 1.094 |
| TestPromptDC | TTA | **0.128** | 0.054 | **1.480** | **0.482** | 5.652 | 1.963 | 2.311 | 0.927 |
| LS Solver | Align | 0.175 | 0.088 | 2.321 | 1.227 | 6.316 | 3.683 | 2.638 | 1.707 |
| Polynomial | Align | 0.169 | 0.084 | 2.255 | 1.168 | 5.183 | 2.672 | 2.337 | 1.306 |
| Piecewise | Align | 0.172 | 0.086 | 2.314 | 1.203 | 5.792 | 3.111 | 2.466 | 1.438 |
| **FoSA (Ours)** | Align | 0.148 | 0.064 | 1.749 | 0.585 | **4.551** | **1.527** | **1.854** | **0.798** |

FoSA is the most accurate sparse depth alignment method on every benchmark. Relative to the LS
solver, it reduces MAE by **27.3%**, **52.3%**, **58.5%**, and **53.3%** on NYUv2, KITTI-DC, DDAD,
and DyDToF. It also outperforms every test-time adaptation method on DDAD and DyDToF without
updating any backbone parameter.

<p align="center">
  <img src="assets/fosa-per-frame-mae.png" alt="Per-frame MAE over 42 frames of NYUv2, DyDToF, KITTI-DC, and DDAD with a UniDepth V2 backbone; FoSA has the lowest MAE in every frame" width="100%">
</p>

Per-frame MAE of the sparse depth alignment methods over 42 frames with a UniDepth V2 backbone.
FoSA attains the lowest MAE in every frame, and its error varies less across frames.

### Generalization across backbones

FoSA attains the lowest MAE in all 15 backbone–dataset configurations and the lowest RMSE in 13 of
them, and it improves on the LS solver in both metrics in every configuration.

<details>
<summary><b>VDA backbone</b></summary>

| Dataset | Sparse pattern | LS RMSE | LS MAE | Poly. RMSE | Poly. MAE | Piece. RMSE | Piece. MAE | **Ours** RMSE | **Ours** MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NYUv2 | Random (500) | 0.279 | 0.161 | 0.237 | 0.135 | 0.271 | 0.162 | **0.171** | **0.079** |
| VOID | 500 | 0.638 | 0.219 | 0.969 | 0.253 | 0.636 | 0.210 | **0.629** | **0.185** |
| DyDToF | Random (500) | 2.934 | 2.011 | 2.475 | 1.369 | **2.312** | 1.286 | 2.336 | **0.904** |
| KITTI-DC | 64-line | 3.657 | 2.063 | 3.911 | 2.277 | 3.468 | 1.767 | **2.243** | **0.752** |
| DDAD | Random (20%) | 11.511 | 7.734 | 17.441 | 12.585 | 8.434 | 5.160 | **5.357** | **1.915** |

</details>

<details>
<summary><b>DVD backbone</b></summary>

| Dataset | Sparse pattern | LS RMSE | LS MAE | Poly. RMSE | Poly. MAE | Piece. RMSE | Piece. MAE | **Ours** RMSE | **Ours** MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NYUv2 | Random (500) | 0.299 | 0.188 | 0.315 | 0.161 | 0.331 | 0.173 | **0.188** | **0.093** |
| VOID | 500 | 0.696 | 0.246 | 1.362 | 0.351 | 0.694 | 0.241 | **0.679** | **0.207** |
| DyDToF | Random (500) | 3.714 | 1.912 | 3.287 | 1.600 | 3.352 | 1.630 | **2.632** | **1.113** |
| KITTI-DC | 64-line | 4.087 | 1.789 | 3.419 | 1.549 | 3.444 | 1.558 | **2.245** | **0.756** |
| DDAD | Random (20%) | 8.492 | 4.750 | 6.505 | 3.713 | 6.221 | 3.533 | **4.957** | **1.825** |

</details>

<details>
<summary><b>DAv3 backbone</b></summary>

| Dataset | Sparse pattern | LS RMSE | LS MAE | Poly. RMSE | Poly. MAE | Piece. RMSE | Piece. MAE | **Ours** RMSE | **Ours** MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NYUv2 | Random (500) | 0.225 | 0.125 | 0.218 | 0.115 | 0.219 | 0.119 | **0.177** | **0.076** |
| VOID | 500 | 0.626 | 0.204 | 1.103 | 0.294 | 0.628 | 0.197 | **0.624** | **0.179** |
| DyDToF | Random (500) | 3.002 | 2.015 | **2.690** | 1.608 | 2.849 | 1.757 | 2.914 | **1.148** |
| KITTI-DC | 64-line | 3.181 | 1.721 | 2.881 | 1.481 | 3.112 | 1.603 | **1.998** | **0.605** |
| DDAD | Random (20%) | 7.885 | 4.805 | 5.805 | 3.251 | 6.509 | 3.737 | **5.217** | **1.709** |

</details>

### Temporal consistency

TEPE on NYUv2 (random 500) and KITTI-DC (64-line) across backbones:

| Dataset | Backbone | LS | Polynomial | Piecewise | **Ours** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **NYUv2** | UniDepth V2 | 0.075 | 0.076 | 0.078 | **0.071** |
|  | VDA | 0.088 | 0.091 | 0.111 | **0.083** |
|  | DVD | 0.106 | 0.105 | 0.112 | **0.094** |
|  | DAv3 | 0.086 | 0.088 | 0.090 | **0.084** |
| **KITTI-DC** | UniDepth V2 | 0.517 | 0.530 | 0.546 | **0.432** |
|  | VDA | 0.517 | 0.958 | 0.543 | **0.464** |
|  | DVD | 0.722 | 0.686 | 0.614 | **0.558** |
|  | DAv3 | 0.600 | 0.602 | 0.639 | **0.437** |

TEPE on DyDToF (random 500) against test-time adaptation:

| Method | Backbone | TEPE |
| :--- | :--- | :---: |
| Zero-DC | Marigold | 1.490 |
| Marigold-DC | Marigold | 1.455 |
| TestPromptDC | UniDepth V2 | 1.070 |
| **FoSA (Ours)** | UniDepth V2 | 0.812 |
| **FoSA (Ours)** | VDA | **0.789** |

<p align="center">
  <img src="assets/fosa-temporal-slices.jpg" alt="Temporal slices on a DyDToF scene: the baselines show vertical streaks where depth jumps between frames, whereas FoSA follows the smooth bands of the ground truth" width="100%">
</p>

Temporal slices on a DyDToF scene (UniDepth V2). Each slice stacks the depth along the red line
over time. The baselines show vertical streaks where the depth jumps between frames, whereas FoSA
follows the smooth bands of the ground truth.

### Robustness to sparser measurements

UniDepth V2 backbone. The same hyperparameters are used at every density.

| Method | KITTI 8-line RMSE | MAE | DyDToF 50 RMSE | MAE | NYUv2 50 RMSE | MAE | VOID 150 RMSE | MAE | DDAD 2% RMSE | MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UniDepth V2 | 2.698 | 1.380 | 5.018 | 3.298 | 0.219 | 0.140 | 0.767 | 0.313 | 6.818 | 3.463 |
| + LS | 2.371 | 1.291 | 2.705 | 1.727 | 0.179 | 0.093 | 0.675 | 0.204 | 6.354 | 3.695 |
| + Poly. | 2.571 | 1.273 | 3.389 | 1.512 | 0.395 | 0.122 | 1.228 | 0.300 | 5.342 | 2.697 |
| + Piece. | 2.398 | 1.280 | **2.642** | **1.494** | 0.183 | 0.097 | 0.675 | 0.201 | 5.696 | 2.944 |
| **+ Ours** | **1.958** | **0.692** | 2.677 | 1.713 | **0.178** | **0.092** | **0.669** | **0.187** | **4.816** | **1.691** |

With 50 points, too few anchors are held out to score a spatial correction, so FoSA falls back to
the global fit and still regularizes it over time with R-TGM.

### Ablation

NYUv2, DVD backbone, 500 random anchors. Components are added one at a time.

| Configuration | RMSE | MAE | TEPE |
| :--- | :---: | :---: | :---: |
| Baseline (LS Solver) | 0.299 | 0.188 | 0.106 |
| + Robust IRLS | 0.285 | 0.173 | 0.099 |
| + Monotonic projection | 0.285 | 0.173 | 0.099 |
| + Spatial field | 0.199 | 0.099 | 0.108 |
| + Bandwidth (density-aware damping) | **0.187** | **0.092** | 0.095 |
| + Auto order | 0.188 | 0.093 | 0.096 |
| + R-TGM (**Full**) | 0.188 | 0.093 | **0.094** |

### Efficiency

KITTI-DC, UniDepth V2 backbone. Memory and time include the backbone; the memory used by the
alignment step alone is in parentheses.

| Method | Required memory (MB) | Inference speed (s/image) |
| :--- | :---: | :---: |
| Zero-DC | 8727 | 20.56 |
| Marigold-DC | 11372 | 21.67 |
| TestPromptDC | 2885 | 5.33 |
| LS Solver | 2636.7 (9.3) | 0.068 |
| Polynomial | 2637.6 (15.3) | 0.069 |
| Piecewise | 2637.6 (16.7) | 0.069 |
| **FoSA (Ours)** | 2637.6 (20.0) | 0.090 |

<p align="center">
  <img src="assets/fosa-teaser.png" alt="Accuracy versus per-frame inference time on KITTI-DC; marker size is memory. FoSA sits with the training-free methods in runtime but close to test-time optimization methods in MAE" width="60%">
</p>

FoSA runs about 60–240× faster than the test-time adaptation methods and processes more than
11 frames per second including the backbone, because each reweighting step is one small linear
solve with no backpropagation through the backbone.

### Qualitative results

<p align="center">
  <img src="assets/fosa-qual-nyuv2.jpg" alt="Qualitative comparison on NYUv2 with a DVD backbone and 500 random anchors: error maps and aligned depth for LS, Polynomial, Piecewise, and FoSA" width="100%">
</p>

NYUv2, DVD backbone, 500 random anchors. In each scene the top row shows the absolute error
(white is low, dark red is high) and the bottom row the aligned depth.

<p align="center">
  <img src="assets/fosa-qual-kitti.jpg" alt="Qualitative comparison on KITTI-DC with a DVD backbone and 64-line LiDAR" width="100%">
</p>

KITTI-DC, DVD backbone, 64-line LiDAR, shown at pixels with valid ground truth.

## Getting started

<!--
TODO(server): Replace this whole section with the real install and run commands, in the same
shape as ReCoG's "Getting started": Python version + CUDA PyTorch note, `pip install -r
requirements.txt`, then 2–4 commented commands (e.g., run a backbone, align one clip with FoSA,
run a baseline, evaluate RMSE/MAE/TEPE). Every command must actually run against the code in this
repository. Point to docs/running.md for details.
-->

The code is being prepared for release. See [docs/methodology.md](docs/methodology.md) for the
full method description.

## Code structure

Coming with the code release.

<!--
TODO(server): Replace with the real tree (ReCoG style: one comment per file, mapped to paper
sections where possible).
-->

## Notes

The numbers above are reported in the paper.
<!-- TODO(server): State plainly what has and has not been re-run from this repository. -->

## Acknowledgments

This research was supported by the Chung-Ang University Research Grants in 2025 and the MSIT
(Ministry of Science and ICT), Korea, under the Graduate School of Virtual Convergence support
program (RS-2024-00418847) supervised by the IITP (Institute for Information & Communications
Technology Planning & Evaluation).
<!-- TODO(server): Credit third-party code actually used (backbone wrappers, GMFlow for TEPE, etc.) and list it in NOTICE.md. -->

## License

This repository is released under the [MIT License](LICENSE). Datasets, backbone code, and model
weights are not redistributed here and remain under their own licenses; see [NOTICE.md](NOTICE.md).
