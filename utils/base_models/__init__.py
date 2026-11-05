"""Frozen depth backbones.

The official code of each backbone is cloned into third_party/ and installed by
scripts/setup_env.sh, one conda environment per backbone; nothing of it is copied
into this repository.

  UniDepthV2  github.com/lpiccinelli-eth/UniDepth              metric depth
  VDA         github.com/DepthAnything/Video-Depth-Anything    metric depth
  DVD         github.com/EnVision-Research/DVD                 affine-invariant disparity
  DAv3        github.com/ByteDance-Seed/Depth-Anything-3       metric depth
"""

import torch.nn.functional as F

BACKBONES = ('UniDepthV2', 'VDA', 'DVD', 'DAv3')


def load_backbone(name, vda_root='', vda_ckpt='', dvd_root='', dvd_ckpt='model.safetensors',
                  dvd_max_side=1280):
    """Frozen backbone on the GPU, tagged with its name so forward_backbone can dispatch."""
    if name == 'UniDepthV2':
        from .unidepth import load_unidepth
        model = load_unidepth()
    elif name == 'VDA':
        from .vda import VDABackbone
        model = VDABackbone(vda_root, vda_ckpt)
    elif name == 'DVD':
        from .dvd import DVDBackbone
        model = DVDBackbone(dvd_root, dvd_ckpt, dvd_max_side)
    elif name == 'DAv3':
        from .da3 import load_da3
        model = load_da3()
    else:
        raise ValueError(f"unknown backbone {name}; expected one of {BACKBONES}")
    model = model.cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.fosa_backbone = name
    return model


def forward_backbone(model, rgb_bt, sparse_bt, amp, T=None):
    """One forward per clip, returning [B*T,1,H,W] at the resolution of sparse_bt.

    rgb_bt / sparse_bt are [B*T,...] with ImageNet-normalised RGB; video backbones
    (VDA, DVD, DAv3) see the clip as one [1,T,...] sequence (batch size is 1 here)."""
    assert rgb_bt.dim() == 4, f"rgb must be [B*T,C,H,W], got {tuple(rgb_bt.shape)}"
    kind = getattr(model, 'fosa_backbone', 'UniDepthV2')
    if kind == 'UniDepthV2':
        from .unidepth import forward_unidepth
        base = forward_unidepth(model, rgb_bt, sparse_bt, amp)
    elif kind == 'DAv3':
        from .da3 import forward_da3
        base = forward_da3(model, rgb_bt, T)
    else:
        T = rgb_bt.shape[0] if T is None else T
        out = model({'rgb': rgb_bt.view(-1, T, *rgb_bt.shape[1:]),
                     'dep': sparse_bt.view(-1, T, *sparse_bt.shape[1:])})['depth']
        base = out.reshape(-1, 1, *out.shape[-2:])
        if getattr(model, 'outputs_disparity', False):
            base = 1.0 / base.float().clamp_min(1e-3 * float(base.max()))
    base = base.float()
    if base.dim() == 3:
        base = base.unsqueeze(1)
    if base.shape[-2:] != sparse_bt.shape[-2:]:
        base = F.interpolate(base, sparse_bt.shape[-2:], mode='bilinear', align_corners=False)
    assert base.shape == sparse_bt.shape, \
        f"backbone returned {tuple(base.shape)}, sparse is {tuple(sparse_bt.shape)}"
    return base
