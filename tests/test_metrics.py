"""
test_metrics.py — RMSE / MAE / TEPE against hand-computed values, and the configs.

    python -m pytest tests -v
"""

import dataclasses
import glob
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils.align import FosaConfig                         # noqa: E402
from utils.clips import crop_box                           # noqa: E402
from utils.config import load_config                       # noqa: E402
from utils.metrics import mae_eval, rmse_eval, tepe_eval   # noqa: E402


def _clip():
    """Two 2x2 frames; gt 0 marks an invalid pixel."""
    gt = torch.tensor([[[1.0, 2.0], [0.0, 4.0]],
                       [[2.0, 0.0], [3.0, 5.0]]]).view(1, 2, 1, 2, 2)
    pred = torch.tensor([[[1.5, 2.0], [9.0, 3.0]],
                         [[2.0, 7.0], [3.5, 6.0]]]).view(1, 2, 1, 2, 2)
    return pred, gt


def test_rmse_and_mae_pool_the_valid_pixels_of_the_clip():
    pred, gt = _clip()
    # valid errors: 0.5, 0, -1 (frame 0); 0, 0.5, 1 (frame 1)
    errs = [0.5, 0.0, 1.0, 0.0, 0.5, 1.0]
    mask = gt > 0
    assert math.isclose(float(rmse_eval(pred, gt, mask)),
                        math.sqrt(sum(e * e for e in errs) / 6), rel_tol=1e-6)
    assert math.isclose(float(mae_eval(pred, gt, mask)), sum(errs) / 6, rel_tol=1e-6)


def test_tepe_on_pixels_valid_in_both_frames():
    pred, gt = _clip()
    # valid in both frames: (0,0) and (1,1)
    #   (0,0): dgt = 2 - 1 = 1,  dpred = 2 - 1.5 = 0.5  -> 0.5
    #   (1,1): dgt = 5 - 4 = 1,  dpred = 6 - 3   = 3    -> 2.0
    assert math.isclose(float(tepe_eval(pred, gt)), (0.5 + 2.0) / 2, rel_tol=1e-6)
    # one frame: no pair, TEPE 0
    assert float(tepe_eval(pred[:, :1], gt[:, :1])) == 0.0


def test_tepe_averages_frame_pairs():
    gt = torch.ones(1, 3, 1, 1, 2)
    pred = torch.tensor([0.0, 0.0, 1.0, 3.0, 1.0, 3.0]).view(1, 3, 1, 1, 2)
    # pair 0->1: |0 - 1|, |0 - 3| -> 2.0 ; pair 1->2: 0, 0 -> 0.0
    assert math.isclose(float(tepe_eval(pred, gt)), 1.0, rel_tol=1e-6)


def test_configs_hold_the_paper_setting():
    """Every config/*.yaml sets every FoSA field to the paper's value (the FosaConfig
    defaults), and only the fit domain differs between backbones."""
    paths = sorted(glob.glob(os.path.join(ROOT, 'config', '*.yaml')))
    assert {os.path.basename(p) for p in paths} == {'unidepth.yaml', 'vda.yaml', 'dvd.yaml',
                                                     'da3.yaml'}
    for p in paths:
        cfg, fcfg = load_config(p)
        want = 'depth' if cfg['backbone'] == 'UniDepthV2' else 'inv'
        assert fcfg == dataclasses.replace(FosaConfig(), domain=want), p
        assert cfg['input_patch_size'] == (0 if cfg['backbone'] == 'UniDepthV2' else 14), p


def test_crop_box_multiple_of_patch():
    assert crop_box(480, 640, 14) == (2, 5, 476, 630)
    assert crop_box(375, 1242, 14) == (5, 5, 364, 1232)
    assert crop_box(100, 10, 14) == (1, 0, 98, 10)              # W < 14: left as is
