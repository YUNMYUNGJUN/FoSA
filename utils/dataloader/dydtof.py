"""DyDToF, video clips.

Layout: one folder per shot directly under --dataset-path,

    <dataset_path>/<shot>/ColorImage/<frame>.jpeg
    <dataset_path>/<shot>/DepthMap/<frame>.npy        dense depth, metres

Frames are resized to height 240 and center-cropped to 228x304. Clips are T
consecutive frames, non-overlapping. Anchors: --num-sparse-samples pixels of the dense
depth per frame, uniformly at random.
"""

import glob
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

HEIGHT = 240
CROP = (228, 304)


class valset(Dataset):
    def __init__(self, args):
        self.T = int(args.num_frames)
        self.num_sample = getattr(args, 'num_sparse_samples', 500)

        data_path = os.path.expanduser(args.dataset_path)
        shots = [s for s in sorted(os.listdir(data_path))
                 if os.path.isdir(os.path.join(data_path, s, 'ColorImage'))]
        if not shots:
            raise FileNotFoundError(f"No <shot>/ColorImage folders found in {data_path}")

        self.video_frame_list = {}
        self.index_map = []                 # (shot, start frame), stride T
        for shot in shots:
            frames = sorted(glob.glob(os.path.join(data_path, shot, 'ColorImage', '*.jpeg')))
            self.video_frame_list[shot] = frames
            for start in range(0, len(frames) - self.T + 1, self.T):
                self.index_map.append((shot, start))

        self.to_tensor = T.Compose([T.Resize(HEIGHT), T.CenterCrop(CROP), T.ToTensor()])
        self.normalize_rgb = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        shot, start = self.index_map[idx]
        frames = self.video_frame_list[shot][start:start + self.T]

        out_rgb, out_dep, out_gt = [], [], []
        for rgb_path in frames:
            rgb = Image.open(rgb_path).convert("RGB")
            depth = np.load(rgb_path.replace('ColorImage', 'DepthMap').replace('.jpeg', '.npy')).astype(np.float32)
            depth = Image.fromarray(depth, mode='F')

            rgb = self.normalize_rgb(self.to_tensor(rgb))
            depth_t = self.to_tensor(depth)  # [1, H, W]

            dep_sp = self.get_random_sparse_depth(depth_t, self.num_sample)

            out_rgb.append(rgb.unsqueeze(0))
            out_dep.append(dep_sp.unsqueeze(0))
            out_gt.append(depth_t.unsqueeze(0))

        return {
            'rgb': torch.cat(out_rgb, dim=0),   # [T, 3, H, W]
            'dep': torch.cat(out_dep, dim=0),   # [T, 1, H, W]
            'gt': torch.cat(out_gt, dim=0),     # [T, 1, H, W]
            'scene_id': shot,
            'idx': idx,
        }

    @staticmethod
    def get_random_sparse_depth(dep, num_sample):
        """num_sample valid pixels of the dense depth, uniformly at random."""
        c, h, w = dep.shape
        assert c == 1
        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)
        if len(idx_nnz) == 0:
            return torch.zeros_like(dep)
        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:min(num_sample, num_idx)]
        selected = idx_nnz[idx_sample]
        mask = torch.zeros_like(dep.view(-1))
        mask[selected] = 1.0
        return dep * mask.view_as(dep)
