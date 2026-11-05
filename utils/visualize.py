"""Visualisation of FoSA results: one PNG per frame,

    RGB           | sparse anchors | backbone depth
    ground truth  | FoSA depth     | |FoSA - GT|

or, without ground truth, RGB | anchors on top and backbone | FoSA below.

Depth uses the Spectral ramp (near = red, far = purple) over the 1st-99th percentile of
the valid ground truth (of the anchors when there is no ground truth), shared by the
anchors, the ground truth and FoSA. The backbone panel gets its own range, since its
output need not be metric. The error uses Reds up to its 99th percentile. Sparse maps
are dilated so that single pixels stay visible.
"""

import os

import numpy as np
from PIL import Image, ImageDraw

_SPECTRAL = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf',
             '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2']
_REDS = ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c',
         '#cb181d', '#a50f15', '#67000d']
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)     # the loaders' ImageNet normalisation
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
GAP = 4


def _lut(stops, n=1024):
    """Piecewise-linear colour table through evenly spaced hex stops, [n, 3] uint8."""
    rgb = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in stops], dtype=np.float64)
    xs = np.linspace(0.0, 1.0, len(stops))
    q = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(q, xs, rgb[:, c]) for c in range(3)], 1).round().astype(np.uint8)


_LUT_DEPTH = _lut(_SPECTRAL)
_LUT_ERR = _lut(_REDS)


def _colorize(x, lo, hi, lut, valid=None):
    """[H,W] float -> [H,W,3] uint8; pixels outside `valid` are black."""
    t = np.clip((x - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    img = lut[np.minimum((t * len(lut)).astype(np.int64), len(lut) - 1)]
    if valid is not None:
        img[~valid] = 0
    return img


def _dilate(img, valid, r):
    """Spread the colour of each valid pixel over its (2r+1)^2 neighbourhood."""
    if r <= 0:
        return img
    H, W = valid.shape
    out = np.zeros_like(img)
    ys, xs = np.nonzero(valid)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out[np.clip(ys + dy, 0, H - 1), np.clip(xs + dx, 0, W - 1)] = img[ys, xs]
    return out


def _tile(img, width, label):
    """Resize to `width` (nearest) and write a label in the top-left corner."""
    h, w = img.shape[:2]
    im = Image.fromarray(img)
    if w != width:
        im = im.resize((width, max(1, round(h * width / w))), Image.NEAREST)
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = d.textbbox((4, 2), label)
    d.rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), fill=(255, 255, 255))
    d.text((4, 2), label, fill=(0, 0, 0))
    return np.asarray(im)


def _grid(rows):
    """Tiles of equal size, rows of equal length, white gaps in between."""
    th, tw = rows[0][0].shape[:2]
    out = np.full((len(rows) * th + (len(rows) - 1) * GAP,
                   len(rows[0]) * tw + (len(rows[0]) - 1) * GAP, 3), 255, dtype=np.uint8)
    for i, row in enumerate(rows):
        for j, t in enumerate(row):
            out[i * (th + GAP):i * (th + GAP) + th, j * (tw + GAP):j * (tw + GAP) + tw] = t
    return out


def frame_panel(rgb, sparse, base, pred, gt=None, max_width=640):
    """rgb [3,H,W] (ImageNet-normalised), sparse / base / pred / gt [H,W] -> uint8 panel.
    gt may be None."""
    W = pred.shape[1]
    width = min(W, max_width)
    r = max(1, int(np.ceil(W / width)))          # dilation that survives the downscale
    rgb8 = (np.clip(rgb.transpose(1, 2, 0) * _STD + _MEAN, 0.0, 1.0) * 255).round().astype(np.uint8)

    sv = sparse > 0
    bv = np.isfinite(base)
    blo, bhi = np.percentile(base[bv], [1, 99]) if bv.any() else (0.0, 1.0)
    if gt is None:
        lo, hi = np.percentile(sparse[sv], [1, 99]) if sv.any() else (0.0, 1.0)
    else:
        gv = (gt > 1e-3) & np.isfinite(gt)
        lo, hi = (np.percentile(gt[gv], [1, 99]) if gv.any() else (0.0, 1.0))
        err = np.abs(pred - gt)
        emax = float(np.percentile(err[gv], 99)) if gv.any() else 1.0
        gt_sparse = gv.mean() < 0.05                  # e.g. projected LiDAR ground truth
        gt_img = _colorize(gt, lo, hi, _LUT_DEPTH, gv)
        err_img = _colorize(err, 0.0, max(emax, 1e-6), _LUT_ERR, gv)
        if gt_sparse:
            gt_img, err_img = _dilate(gt_img, gv, r), _dilate(err_img, gv, r)
    pred_img = _tile(_colorize(pred, lo, hi, _LUT_DEPTH, np.isfinite(pred)), width, 'FoSA')
    rgb_tile = _tile(rgb8, width, 'RGB')
    sparse_tile = _tile(_dilate(_colorize(sparse, lo, hi, _LUT_DEPTH, sv), sv, r), width, 'anchors')
    base_tile = _tile(_colorize(base, blo, bhi, _LUT_DEPTH, bv), width, 'backbone')
    if gt is None:
        return _grid([[rgb_tile, sparse_tile], [base_tile, pred_img]])
    return _grid([[rgb_tile, sparse_tile, base_tile],
                  [_tile(gt_img, width, 'GT'), pred_img,
                   _tile(err_img, width, f'|FoSA - GT|, max {emax:.2f} m')]])


def save_clip(out_dir, idx, rgb, sparse, base, pred, gt=None, max_width=640):
    """Write <out_dir>/<idx>_<t>.png for every frame of one clip ([T,C,H,W] numpy arrays;
    gt may be None)."""
    os.makedirs(out_dir, exist_ok=True)
    for t in range(pred.shape[0]):
        panel = frame_panel(rgb[t], sparse[t, 0], base[t, 0], pred[t, 0],
                            None if gt is None else gt[t, 0], max_width)
        Image.fromarray(panel).save(os.path.join(out_dir, f'{idx:05d}_{t}.png'))
