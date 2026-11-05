"""NYUv2 video clips.

Layout: <dataset_path>/<scene>/<frame>.h5, each file holding 'rgb' [3,H,W] uint8 and
'depth' [H,W] float (metres). Clips are T consecutive frames, non-overlapping.
Anchors: --num-sparse-samples pixels of the dense depth per frame, uniformly at random.
"""

import os
import os.path as osp
from glob import glob

import h5py
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


def _make_sparse_from_dense(depth: torch.Tensor, num_samples: int = 500) -> torch.Tensor:
    """num_samples valid pixels of the dense depth, uniformly at random."""
    # depth: [1, H, W]
    c, h, w = depth.shape
    assert c == 1
    idx_nnz = torch.nonzero(depth.reshape(-1) > 0.0001, as_tuple=False).reshape(-1)
    if idx_nnz.numel() == 0:
        return torch.zeros_like(depth)
    num = min(num_samples, idx_nnz.numel())
    perm = torch.randperm(idx_nnz.numel())[:num]
    chosen = idx_nnz[perm]
    mask = torch.zeros((h * w), dtype=depth.dtype, device=depth.device)
    mask[chosen] = 1.0
    mask = mask.view(1, h, w)
    return depth * mask


class valset(Dataset):
    def __init__(self, args):
        self.datapath = os.path.expanduser(args.dataset_path)
        self.T = int(args.num_frames)
        self.num_sparse = getattr(args, 'num_sparse_samples', 500)
        assert self.T > 0, "num_frames must be > 0"

        scene_dirs = sorted(d for d in os.listdir(self.datapath)
                            if osp.isdir(osp.join(self.datapath, d)))
        if not scene_dirs:
            raise ValueError(f"No scene directories found in {self.datapath}")

        self.files = {}
        self.clips = []                     # (scene_id, start frame), stride T
        for scene_id in scene_dirs:
            h5_files = sorted(glob(osp.join(self.datapath, scene_id, '*.h5')))
            if len(h5_files) < self.T:
                continue
            self.files[scene_id] = h5_files
            for start in range(0, len(h5_files) - self.T + 1, self.T):
                self.clips.append((scene_id, start))
        if not self.clips:
            raise ValueError(f"No clips of {self.T} frames found in {self.datapath}")
        print(f"NYUv2: {len(self.files)} scenes, {len(self.clips)} clips")

        # NYUv2 nominal intrinsics (constant for all frames)
        self.K = torch.tensor([
            5.1885790117450188e+02, 0.0, 3.2558244941119034e+02,
            0.0, 5.1946961112127485e+02, 2.5373616633400465e+02,
            0.0, 0.0, 1.0
        ], dtype=torch.float32).reshape(3, 3)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        sid, start = self.clips[idx]
        rgbs, deps, gts = [], [], []
        for t in range(self.T):
            with h5py.File(self.files[sid][start + t], 'r') as f:
                rgb_data = f['rgb'][:]      # [3, H, W]
                depth_data = f['depth'][:]  # [H, W]

            rgb_tensor = torch.from_numpy(rgb_data).float() / 255.0
            rgb_tensor = TF.normalize(rgb_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            gt_tensor = torch.from_numpy(depth_data).float().unsqueeze(0)  # [1, H, W]

            dep_tensor = _make_sparse_from_dense(gt_tensor.clone(), num_samples=self.num_sparse)

            rgbs.append(rgb_tensor)
            gts.append(gt_tensor)
            deps.append(dep_tensor)

        return {
            'rgb': torch.stack(rgbs, dim=0).float(),    # [T, 3, H, W]
            'dep': torch.stack(deps, dim=0).float(),    # [T, 1, H, W]
            'gt': torch.stack(gts, dim=0).float(),      # [T, 1, H, W]
            'intrinsic': torch.stack([self.K.clone()] * self.T, dim=0),
            'idx': idx,
            'scene_id': sid,
        }
