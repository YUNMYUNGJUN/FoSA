<div align="center">
<h1 style="border-bottom: none; margin-bottom: 0px">FoSA: Fast Video Depth Completion via Fourier Scale-and-Shift Alignment</h1>

[**Myungjun Yun**](https://github.com/YUNMYUNGJUN) · [**Jin-Hwi Park**](https://www.csilab.cau.ac.kr/home)<sup>&dagger;</sup>

[CSILab](https://www.csilab.cau.ac.kr/home), <img src="assets/cau-logo.png" alt="Chung-Ang University" height="18" align="absmiddle"> [Chung-Ang University](https://www.cau.ac.kr)

&dagger;Corresponding author

<img src='https://img.shields.io/badge/Paper-Coming_Soon-lightgrey' alt='Paper coming soon'>
<a href="LICENSE"><img src='https://img.shields.io/badge/License-MIT-yellow' alt='License: MIT'></a>

</div>

This work presents **FoSA**, a training-free method that turns the prediction of a frozen depth
foundation model into accurate, temporally stable metric video depth using sparse range
measurements. Each frame is aligned by one closed-form linear solve, with no backpropagation:

- 〰️ **Fourier scale-and-shift fields** replace the global affine fit, so the correction can vary across the image.
- 📡 **Density-aware damping** attenuates each frequency by the local spacing of the measurements, so the degrees of freedom follow their Nyquist limit without manual tuning.
- 🎞️ **Residual Temporal Gradient Matching (R-TGM)** regularizes the correction residual over time, keeping the backbone's own temporal behavior without optical flow or camera pose.

🏆 With a UniDepth V2 backbone, FoSA reduces the MAE of the least-squares fit by 27–59% on NYUv2,
KITTI-DC, DDAD, and DyDToF. It also outperforms test-time adaptation methods on DDAD and DyDToF while
running 60–240× faster than them (0.09 s per frame, backbone included).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/fosa-teaser-dark.png">
    <img src="assets/fosa-teaser-light.png" alt="MAE on KITTI-DC versus inference time per frame (log scale), bubble size is GPU memory: training-free FoSA reaches 0.585 m at 0.09 s, close to test-time optimization methods (0.48-0.55 m at 5-22 s) and far below LS, Polynomial, and Piecewise alignment (about 1.2 m at 0.07 s)" width="85%">
  </picture>
</p>
<p align="center">
  <img src="assets/fosa-overview.png" alt="FoSA overview: Stage 1 expands scale and shift into damped Fourier fields; Stage 2 (R-TGM) regularizes the correction residual over time; each frame is aligned by a closed-form linear solve" width="100%">
</p>

## 📰 News

- **05-11-2026:** 🚀 Project page and foundation-model setup released. Code coming soon.

## 🧩 Supported Foundation Models

FoSA aligns the output of a frozen foundation model, so the same alignment runs on single-image and
video backbones.

<div align="center">

| 🗃️ Foundation model | 🎞️ Input | 🐍 conda env | 🔗 Official code |
| :--- | :---: | :---: | :--- |
| UniDepth V2 | Single image | `fosa-unidepth` | [lpiccinelli-eth/UniDepth](https://github.com/lpiccinelli-eth/UniDepth) |
| Video Depth Anything | Video | `fosa-vda` | [DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) |
| DVD | Video | `fosa-dvd` | [EnVision-Research/DVD](https://github.com/EnVision-Research/DVD) |
| Depth Anything 3 | 1–N images | `fosa-da3` | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) |

</div>

## 🚀 Quick Start

### 📦 Setup

Each foundation model requires its own conda environment due to dependency conflicts (torch,
xformers, and CUDA builds differ). Install [Anaconda or Miniconda](https://docs.anaconda.com/miniconda/) first.

```bash
git clone https://github.com/YUNMYUNGJUN/FoSA.git
cd FoSA

bash scripts/setup_env.sh all         # set up all four foundation models
# Or set up individual foundation models
# bash scripts/setup_env.sh unidepth  # set up UniDepth V2 env
# bash scripts/setup_env.sh vda       # set up Video Depth Anything env
# bash scripts/setup_env.sh dvd       # set up DVD env
# bash scripts/setup_env.sh da3       # set up Depth Anything 3 env
```

Each command does the following:

1. Clones the model's official repository into `third_party/` at a pinned commit.
2. Creates a conda environment.
3. Installs the model following its official instructions.
4. Downloads the checkpoints that are not fetched automatically (Video Depth Anything, DVD).
   UniDepth V2 and Depth Anything 3 load their weights from Hugging Face on first use.

After setup, activate the corresponding environment:

```bash
conda activate fosa-unidepth  # for UniDepth V2
conda activate fosa-vda       # for Video Depth Anything
conda activate fosa-dvd       # for DVD
conda activate fosa-da3       # for Depth Anything 3
```

### 💻 Usage

Coming with the code release.

<!--
TODO(server): Replace with the real commands in the CAPA style: one bash block with a comment per
command (e.g., align a single clip, process a directory, evaluate RMSE/MAE/TEPE), each run inside
one of the conda environments above. Every command must actually run against this repository.
-->

## 📊 Results

UniDepth V2 backbone for every sparse depth alignment method. **TTA**: test-time adaptation of the
backbone. **Align**: sparse depth alignment of a frozen backbone's output. Metrics are in meters,
and **bold** marks the best value in each column.

<div align="center">

<table>
<thead>
<tr><th rowspan="2" align="left">Method</th><th rowspan="2" align="left">Type</th><th colspan="2">NYUv2</th><th colspan="2">KITTI-DC</th><th colspan="2">DDAD</th><th colspan="2">DyDToF</th></tr>
<tr><th>RMSE</th><th>MAE</th><th>RMSE</th><th>MAE</th><th>RMSE</th><th>MAE</th><th>RMSE</th><th>MAE</th></tr>
</thead>
<tbody>
<tr><td align="left">Zero-DC</td><td align="left">TTA</td><td align="center">0.152</td><td align="center">0.097</td><td align="center">1.515</td><td align="center">0.509</td><td align="center">7.163</td><td align="center">3.621</td><td align="center">3.399</td><td align="center">2.038</td></tr>
<tr><td align="left">Marigold-DC</td><td align="left">TTA</td><td align="center">0.135</td><td align="center"><b>0.053</b></td><td align="center">1.483</td><td align="center">0.550</td><td align="center">6.401</td><td align="center">2.708</td><td align="center">2.662</td><td align="center">1.094</td></tr>
<tr><td align="left">TestPromptDC</td><td align="left">TTA</td><td align="center"><b>0.128</b></td><td align="center">0.054</td><td align="center"><b>1.480</b></td><td align="center"><b>0.482</b></td><td align="center">5.652</td><td align="center">1.963</td><td align="center">2.311</td><td align="center">0.927</td></tr>
<tr><td align="left">LS Solver</td><td align="left">Align</td><td align="center">0.175</td><td align="center">0.088</td><td align="center">2.321</td><td align="center">1.227</td><td align="center">6.316</td><td align="center">3.683</td><td align="center">2.638</td><td align="center">1.707</td></tr>
<tr><td align="left">Polynomial</td><td align="left">Align</td><td align="center">0.169</td><td align="center">0.084</td><td align="center">2.255</td><td align="center">1.168</td><td align="center">5.183</td><td align="center">2.672</td><td align="center">2.337</td><td align="center">1.306</td></tr>
<tr><td align="left">Piecewise</td><td align="left">Align</td><td align="center">0.172</td><td align="center">0.086</td><td align="center">2.314</td><td align="center">1.203</td><td align="center">5.792</td><td align="center">3.111</td><td align="center">2.466</td><td align="center">1.438</td></tr>
<tr><td align="left"><b>FoSA (Ours)</b></td><td align="left">Align</td><td align="center">0.148</td><td align="center">0.064</td><td align="center">1.749</td><td align="center">0.585</td><td align="center"><b>4.551</b></td><td align="center"><b>1.527</b></td><td align="center"><b>1.854</b></td><td align="center"><b>0.798</b></td></tr>
</tbody>
</table>

</div>

<p align="center">
  <img src="assets/fosa-qual-kitti.jpg" alt="Qualitative comparison on KITTI-DC with a DVD backbone: error maps and aligned depth for LS, Polynomial, Piecewise, and FoSA" width="100%">
</p>

For the other backbones (VDA, DVD, DAv3), temporal consistency (TEPE), sparser measurements,
ablation, efficiency, and qualitative results, see [docs/RESULTS.md](docs/RESULTS.md).

## 📚 Useful Documentation

- 🧮 [Method](docs/METHOD.md): the alignment, equation by equation
- 📊 [Results](docs/RESULTS.md): all tables and figures of the paper

## 📄 License

Copyright © 2026 YUNMYUNGJUN.

The FoSA source code is released under the [MIT License](LICENSE).

## 🙏 Acknowledgments

This research was supported by the Chung-Ang University Research Grants in 2025 and the MSIT
(Ministry of Science and ICT), Korea, under the Graduate School of Virtual Convergence support
program (RS-2024-00418847) supervised by the IITP (Institute for Information & Communications
Technology Planning & Evaluation).
