# Attribution

This repository contains source code, documentation, and figures from the FoSA paper
(`assets/fosa-*.png`, `assets/fosa-*.jpg`), and the Chung-Ang University logo in `assets/`, which
belongs to its owner. It does not redistribute datasets, backbone model weights, generated
predictions, experiment outputs, or research manuscripts.

The benchmarks (NYUv2, VOID, DyDToF, KITTI-DC, DDAD) and the depth backbones (UniDepth V2, Video
Depth Anything, DVD, Depth Anything 3) are obtained from their original sources and are subject to
their respective licenses.

## Third-party code

No third-party source code is copied into this repository. The backbone wrappers in
`utils/base_models/` import the official code of each foundation model, which
`scripts/setup_env.sh` clones into `third_party/` (not tracked); the README lists their sources,
pinned commits, and licenses under "Third-Party Components".

The following third-party values appear in the code:

- `utils/visualize.py` uses the color values of the ColorBrewer "Spectral" (11 classes) and "Reds"
  (9 classes) schemes. ColorBrewer: Copyright (c) 2002 Cynthia Brewer, Mark Harrower, and The
  Pennsylvania State University. Licensed under the Apache License, Version 2.0.
- `utils/order.py` splits the held-out anchors with a hash that uses the multiplier constants of
  the MurmurHash3 32-bit finalizer (Austin Appleby, public domain) and the spatial-hash primes of
  Teschner et al., "Optimized Spatial Hashing for Collision Detection of Deformable Objects"
  (VMV 2003).
