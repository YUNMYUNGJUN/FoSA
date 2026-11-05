"""Everything main.py uses.

    align.py        FoSA for one clip: FosaConfig, collect_seeds, fit, predict
    basis.py        Fourier basis and frequency set F_U        (Sec. IV-A)
    bandwidth.py    density-aware damping, Delta(p)            (Sec. IV-B)
    temporal.py     R-TGM                                      (Sec. IV-C)
    solver.py       normal equations, Huber IRLS, projection   (Sec. IV-D)
    order.py        held-out order selection                   (Sec. IV-E)
    metrics.py      RMSE, MAE, TEPE
    visualize.py    per-frame result panels
    config.py       config/<backbone>.yaml
    clips.py        input sequences, cropping, the backbone cache
    base_models/    UniDepth V2, Video Depth Anything, DVD, Depth Anything 3
    dataloader/     NYUv2, VOID, KITTI-DC, DyDToF, DDAD video clips
"""

from .align import FosaConfig, FosaState, collect_seeds, fit, predict

__all__ = ['FosaConfig', 'FosaState', 'collect_seeds', 'fit', 'predict']
