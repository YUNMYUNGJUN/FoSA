"""Clip handling for main.py: input sequences, cropping, anchor thinning, the backbone cache."""

import glob
import json
import os

import torch

from .dataloader.seeded import clip_seed

_MEAN = (0.485, 0.456, 0.406)       # the ImageNet normalisation of the dataset loaders
_STD = (0.229, 0.224, 0.225)


def sequence_files(path):
    """A .pt file, or every *.pt file under a folder (sorted)."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '**', '*.pt'), recursive=True))
        if not files:
            raise FileNotFoundError(f"no .pt files under {path}")
        return files
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return [path]


def load_sequence(path):
    """One sequence saved with torch.save as a dict of tensors:

        rgb     [N,3,H,W] uint8 (or float in [0,1])
        sparse  [N,H,W] float, metres, 0 = no measurement
        gt      [N,H,W] float, metres, 0 = invalid          (optional)

    Returns rgb ImageNet-normalised [N,3,H,W] and sparse / gt as [N,1,H,W] float32.
    """
    d = torch.load(path, map_location='cpu', weights_only=True)
    for k in ('rgb', 'sparse'):
        if k not in d:
            raise KeyError(f"{path}: missing '{k}'")
    rgb = d['rgb']
    if rgb.dim() != 4 or rgb.shape[1] != 3:
        raise ValueError(f"{path}: rgb must be [N,3,H,W], got {tuple(rgb.shape)}")
    rgb = rgb.float().div(255) if rgb.dtype == torch.uint8 else rgb.float()
    mean = torch.tensor(_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_STD).view(1, 3, 1, 1)
    rgb = (rgb - mean) / std

    def frames(x, name):
        x = x.float()
        x = x.unsqueeze(1) if x.dim() == 3 else x
        if x.shape != (rgb.shape[0], 1, *rgb.shape[-2:]):
            raise ValueError(f"{path}: {name} must be [N,H,W] matching rgb, got {tuple(x.shape)}")
        return x

    gt = d.get('gt')
    return rgb, frames(d['sparse'], 'sparse'), None if gt is None else frames(gt, 'gt')


def crop_box(H, W, ps):
    """(y0, x0, h, w) of the center crop of an H x W image to multiples of ps."""
    h, w = (H // ps) * ps or H, (W // ps) * ps or W
    return (H - h) // 2, (W - w) // 2, h, w


def center_crop(x, ps):
    """Center-crop [..., H, W] to multiples of ps."""
    y0, x0, h, w = crop_box(*x.shape[-2:], ps)
    return x[..., y0:y0 + h, x0:x0 + w]


def thin_anchors(sparse, keep, base_seed, idx):
    """Keep each anchor with probability `keep`, from a generator seeded by the clip alone."""
    if keep >= 1.0:
        return sparse
    g = torch.Generator().manual_seed(clip_seed(base_seed, idx) ^ 0x5EED)
    m = torch.rand(sparse.shape, generator=g) < float(keep)
    return sparse * m.to(sparse.device, sparse.dtype)


class WithCache(torch.utils.data.Dataset):
    """Adds the cached backbone output of each clip ('base'), read in the loader workers
    so that it overlaps with the GPU work of the previous clip."""

    def __init__(self, dataset, cache_dir):
        self.dataset = dataset
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        path = os.path.join(self.cache_dir, f'{idx:05d}.pt') if self.cache_dir else ''
        if path and os.path.exists(path):
            sample['base'] = torch.load(path, map_location='cpu', weights_only=True)
        return sample


def check_cache(cache_dir, meta):
    """The cache holds backbone outputs only; refuse one built for another setting."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, 'meta.json')
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        bad = [k for k in meta if old.get(k) != meta[k]]
        if bad:
            raise RuntimeError(f"--base-cache {cache_dir} was built for a different "
                               f"setting ({bad}); use another folder")
    else:
        with open(path, 'w') as f:
            json.dump(meta, f, indent=1)
