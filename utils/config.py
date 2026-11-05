"""Run configuration: one YAML file per backbone in config/, holding the paper's settings.

    backbone: UniDepthV2 | VDA | DVD | DAv3
    amp: bf16                 # UniDepth V2 autocast (off | fp16 | bf16)
    input_patch_size: 0       # center-crop the inputs to multiples of this
    vda_root / vda_ckpt / dvd_root / dvd_ckpt / dvd_max_side   # VDA and DVD only
    fosa:                     # utils.align.FosaConfig fields
      domain: depth
      ...

Relative paths are taken from the repository root.
"""

import dataclasses
import os

import yaml

from .align import FosaConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULTS = {'backbone': None, 'amp': 'bf16', 'input_patch_size': 0,
            'vda_root': '', 'vda_ckpt': '', 'dvd_root': '', 'dvd_ckpt': 'model.safetensors',
            'dvd_max_side': 1280}
_PATHS = ('vda_root', 'vda_ckpt', 'dvd_root')


def load_config(path, fosa_overrides=None):
    """-> (backbone settings dict, FosaConfig). fosa_overrides: {field: value}, None = keep."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    fosa = dict(raw.pop('fosa', None) or {})
    unknown = sorted(set(raw) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}")
    cfg = {**DEFAULTS, **raw}
    if not cfg['backbone']:
        raise ValueError(f"{path}: 'backbone' is required")
    for k in _PATHS:
        if cfg[k] and not os.path.isabs(cfg[k]):
            cfg[k] = os.path.join(ROOT, cfg[k])
    if cfg['backbone'] == 'VDA' and not cfg['vda_ckpt'] and cfg['vda_root']:
        cfg['vda_ckpt'] = os.path.join(cfg['vda_root'], 'checkpoints',
                                       'metric_video_depth_anything_vitl.pth')

    fields = {f.name: f for f in dataclasses.fields(FosaConfig)}
    unknown = sorted(set(fosa) - set(fields))
    if unknown:
        raise ValueError(f"{path}: unknown fosa keys {unknown}")
    fosa.update({k: v for k, v in (fosa_overrides or {}).items() if v is not None})
    # YAML reads 1e-2 as a string; cast every value to the type of its default
    fosa = {k: type(fields[k].default)(v) for k, v in fosa.items()}
    return cfg, FosaConfig(**fosa)
