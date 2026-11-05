"""Depth Anything 3 (DA3METRIC-LARGE) as a frozen backbone; installed from
third_party/Depth-Anything-3. Its internal model is called on the normalised clip."""

import torch


def load_da3():
    from depth_anything_3.api import DepthAnything3
    return DepthAnything3.from_pretrained("depth-anything/da3metric-large")


def forward_da3(model, rgb_bt, T=None):
    """[B*T,3,H,W] normalised RGB -> [B*T,1,H',W'] metric depth."""
    T = rgb_bt.shape[0] if T is None else T
    with torch.autocast('cuda', dtype=torch.float16):
        raw = model.model(rgb_bt.view(-1, T, *rgb_bt.shape[1:]), None, None, [], False, False,
                          'saddle_balanced')
    d = raw['depth'] if isinstance(raw, dict) else raw.depth
    if d.dim() == 5 and d.shape[-1] == 1:
        d = d.permute(0, 1, 4, 2, 3)
    elif d.dim() == 4:
        d = d.unsqueeze(2)
    return d.reshape(-1, 1, *d.shape[-2:])
