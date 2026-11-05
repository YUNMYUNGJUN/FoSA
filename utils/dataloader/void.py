"""VOID (Wong et al., RA-L 2020), video clips.

--dataset-path is a density folder of the VOID release, e.g. .../void_release/void_500 or
.../void_150. It must hold test_image.txt, test_sparse_depth.txt and test_ground_truth.txt;
their entries ('void_500/data/<scene>/image/<t>.png', ...) are relative to the parent of
that folder. Frames are grouped by scene, sorted by file name, and cut into
non-overlapping clips of T frames.
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_depth(path):
    """uint16 PNG / 256 -> metres (from the VOID release tools)."""
    z = np.array(Image.open(path), dtype=np.float32)
    z = z / 256.0
    z[z <= 0] = 0.0
    return z


def _read_list(path, root):
    with open(path, 'r') as f:
        return [os.path.join(root, line.strip()) for line in f if line.strip()]


class valset(Dataset):
    def __init__(self, args):
        self.T = int(args.num_frames)
        assert self.T > 0, 'num_frames must be > 0'
        density_dir = os.path.normpath(os.path.expanduser(args.dataset_path))
        root_dir = os.path.dirname(density_dir)

        rgb_paths = _read_list(os.path.join(density_dir, 'test_image.txt'), root_dir)
        dep_paths = _read_list(os.path.join(density_dir, 'test_sparse_depth.txt'), root_dir)
        gt_paths = _read_list(os.path.join(density_dir, 'test_ground_truth.txt'), root_dir)
        assert len(rgb_paths) == len(dep_paths) == len(gt_paths), 'list files differ in length'

        # scene id = the folder above image/, e.g. .../data/stairs4/image/<t>.png -> stairs4
        scenes = {}
        for rgb, dep, gt in zip(rgb_paths, dep_paths, gt_paths):
            scene_id = os.path.basename(os.path.dirname(os.path.dirname(rgb)))
            scenes.setdefault(scene_id, []).append({'rgb': rgb, 'dep': dep, 'gt': gt})
        for sid in scenes:
            scenes[sid].sort(key=lambda r: os.path.basename(r['rgb']))

        self.clips = []                     # (scene_id, start frame), stride T
        for sid in sorted(scenes.keys()):
            for start in range(0, len(scenes[sid]) - self.T + 1, self.T):
                self.clips.append((sid, start))
        if not self.clips:
            raise ValueError(f'No clips of {self.T} frames found in {density_dir}')
        self._scenes = scenes
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        sid, start = self.clips[idx]
        rgbs, deps, gts = [], [], []
        for rec in self._scenes[sid][start:start + self.T]:
            rgb_np = np.array(Image.open(rec['rgb']).convert('RGB'), dtype=np.float32) / 255.0
            rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).float()
            rgbs.append((rgb_t - self.mean) / self.std)
            gts.append(torch.from_numpy(load_depth(rec['gt'])[None, ...]).float())
            deps.append(torch.from_numpy(load_depth(rec['dep'])[None, ...]).float())

        return {
            'rgb': torch.stack(rgbs, dim=0),    # [T,3,H,W]
            'dep': torch.stack(deps, dim=0),    # [T,1,H,W]
            'gt': torch.stack(gts, dim=0),      # [T,1,H,W]
            'intrinsic': torch.eye(3, dtype=torch.float32).expand(self.T, 3, 3).clone(),
            'idx': idx,
            'scene_id': sid,
        }
