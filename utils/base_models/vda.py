"""Video Depth Anything (ViT-L) as a frozen backbone.

Uses the official clone in third_party/Video-Depth-Anything (``vda_root`` in the
config) and a checkpoint from its get_weights.sh. The metric checkpoint
(metric_video_depth_anything_vitl.pth, the paper's) predicts depth; any other
checkpoint is taken to predict relative disparity, and its inverse is handed to FoSA so
that the inverse-depth fit sees the disparity itself.

Each clip is resized to 518x518, run through the network as one sequence, and the
output is resized back to the input resolution.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = 518


def _is_utils(name):
    return name == 'utils' or name.startswith('utils.')


def _import_vda(vda_root):
    """VideoDepthAnything from the official clone.

    VDA's video_depth.py runs ``from utils.util import ...`` against the top-level
    ``utils`` package of its own repository, which this repository's ``utils`` shadows.
    Our modules are set aside while VDA is imported and restored afterwards; VDA keeps
    the functions it imported.
    """
    ours = {k: m for k, m in sys.modules.items() if _is_utils(k)}
    for k in ours:
        del sys.modules[k]
    sys.path.insert(0, vda_root)
    try:
        from video_depth_anything.video_depth import VideoDepthAnything
    finally:
        sys.path.remove(vda_root)
        for k in [k for k in sys.modules if _is_utils(k)]:
            del sys.modules[k]
        sys.modules.update(ours)
    return VideoDepthAnything


class VDABackbone(nn.Module):
    def __init__(self, vda_root, ckpt=''):
        super().__init__()
        vda_root = os.path.abspath(os.path.expanduser(vda_root or ''))
        if not os.path.isdir(os.path.join(vda_root, 'video_depth_anything')):
            raise ImportError(f"VDA backbone: vda_root={vda_root} is not a clone of "
                              "github.com/DepthAnything/Video-Depth-Anything "
                              "(run: bash scripts/setup_env.sh vda)")
        VideoDepthAnything = _import_vda(vda_root)
        if not ckpt:
            ckpt = os.path.join(vda_root, 'checkpoints', 'metric_video_depth_anything_vitl.pth')
        metric = 'metric' in os.path.basename(ckpt)
        self.net = VideoDepthAnything(encoder='vitl', features=256,
                                      out_channels=[256, 512, 1024, 1024], metric=metric)
        self.net.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
        self.outputs_disparity = not metric

    @torch.no_grad()
    def forward(self, sample):
        frames = sample['rgb']                                  # [B,T,3,H,W], normalised
        B, T, C, H, W = frames.shape
        x = F.interpolate(frames.reshape(B * T, C, H, W), (INPUT_SIZE, INPUT_SIZE),
                          mode='bilinear', align_corners=False, antialias=False)
        depth = self.net(x.view(B, T, C, INPUT_SIZE, INPUT_SIZE)).to(frames.dtype)  # [B,T,518,518]
        depth = F.interpolate(depth.flatten(0, 1).unsqueeze(1), size=(H, W),
                              mode='bilinear', align_corners=True)
        return {'depth': depth.view(B, T, 1, H, W)}
