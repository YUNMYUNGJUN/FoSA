"""KITTI depth completion, video clips.

Layout (one folder per drive, file names = frame index):

    <dataset_path>/val/<drive>/rgb/<frame>.png         RGB
    <dataset_path>/val/<drive>/depth/<frame>.png       dense GT, uint16 / 256 = metres
    <dataset_path>/val/<drive>/lidar/<frame>.png       sparse LiDAR, uint16 / 256 = metres
    <dataset_path>/val/<drive>/intrinsics/<frame>.txt  3x3 K (only needed for --num-lidar-lines < 64)

Clips are T consecutive frames, non-overlapping. Frames are resized to 375x1242 and
depths are clipped to 80 m.
"""

import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from .lidar import sample_lidar_lines

HEIGHT, WIDTH = 375, 1242
MAX_DEPTH = 80.0
DEFAULT_K = np.array([[721.5377, 0.0, 609.5593],
                      [0.0, 721.5377, 172.8540],
                      [0.0, 0.0, 1.0]])


def read_depth(file_name, max_depth=None):
    """uint16 PNG / 256 -> metres, optionally clipped to max_depth."""
    assert os.path.exists(file_name), f"file not found: {file_name}"
    image_depth = np.array(Image.open(file_name)).astype(np.float32) / 256.0
    if max_depth is not None:
        image_depth = np.clip(image_depth, 0, max_depth)
    return image_depth


def read_intrinsics(intrinsics_file):
    """3x3 K from 9 numbers (row-major), three lines of three, or 'fx fy cx cy'."""
    with open(intrinsics_file, 'r') as f:
        content = f.read().strip()
    values = [float(x) for x in content.split()]
    if len(values) == 9:
        return np.array(values).reshape(3, 3)
    if len(values) == 4:
        fx, fy, cx, cy = values
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    raise ValueError(f"Could not parse intrinsics from {intrinsics_file} ({len(values)} values)")


class valset(Dataset):
    def __init__(self, args, mode='val'):
        self.root_dir = os.path.join(os.path.expanduser(args.dataset_path), mode)
        self.T = int(args.num_frames)
        self.num_lidar_lines = int(getattr(args, 'num_lidar_lines', 64))

        self.scene_dirs = sorted(
            os.path.join(self.root_dir, s) for s in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, s)))
        if not self.scene_dirs:
            raise FileNotFoundError(f"No scene directories found in {self.root_dir}")

        self.index_map = []            # (scene_idx, start frame), stride T
        self.scenes_frame_names = []
        for scene_idx, scene_dir in enumerate(self.scene_dirs):
            rgb_dir = os.path.join(scene_dir, 'rgb')
            frame_names = sorted(os.path.splitext(f)[0] for f in os.listdir(rgb_dir) if f.endswith('.png'))
            self.scenes_frame_names.append(frame_names)
            for start_idx in range(0, len(frame_names) - self.T + 1, self.T):
                self.index_map.append((scene_idx, start_idx))
        if not self.index_map:
            raise ValueError(f"No clips of {self.T} frames found in {self.root_dir}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        scene_idx, start = self.index_map[idx]
        scene_dir = self.scene_dirs[scene_idx]
        frame_names = self.scenes_frame_names[scene_idx][start:start + self.T]

        rgbs, depths, lidars, intrinsics = [], [], [], []
        for fname in frame_names:
            rgb = Image.open(os.path.join(scene_dir, 'rgb', f"{fname}.png")).convert("RGB")
            depth = read_depth(os.path.join(scene_dir, 'depth', f"{fname}.png"), max_depth=MAX_DEPTH)
            lidar = read_depth(os.path.join(scene_dir, 'lidar', f"{fname}.png"), max_depth=MAX_DEPTH)

            k_path = os.path.join(scene_dir, 'intrinsics', f"{fname}.txt")
            if os.path.exists(k_path):
                K = read_intrinsics(k_path)
            else:
                print(f"Warning: intrinsics not found for {fname}, using default KITTI values")
                K = DEFAULT_K

            # keep a subset of the 64 LiDAR lines (--num-lidar-lines)
            keep_ratio = float(self.num_lidar_lines) / 64.0
            assert 0 <= keep_ratio <= 1.0, keep_ratio
            if keep_ratio >= 0.9999:
                sparse_lidar = lidar
            elif keep_ratio > 0:
                sparse_lidar = sample_lidar_lines(lidar[:, :, None], K, keep_ratio=keep_ratio)[:, :, 0]
            else:
                sparse_lidar = np.zeros_like(lidar)

            rgb = TF.resize(rgb, (HEIGHT, WIDTH), interpolation=Image.BICUBIC)
            rgb = TF.to_tensor(rgb)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

            depth = Image.fromarray(depth.astype(np.float32), mode="F")
            depth = TF.resize(depth, (HEIGHT, WIDTH), interpolation=Image.NEAREST)
            depth = TF.to_tensor(np.array(depth))

            lidar_img = Image.fromarray(sparse_lidar.astype(np.float32), mode="F")
            lidar_img = TF.resize(lidar_img, (HEIGHT, WIDTH), interpolation=Image.NEAREST)
            lidar_img = TF.to_tensor(np.array(lidar_img))

            rgbs.append(rgb.unsqueeze(0))
            depths.append(depth.unsqueeze(0))
            lidars.append(lidar_img.unsqueeze(0))
            intrinsics.append(torch.from_numpy(K).float().unsqueeze(0))

        return {
            "rgb": torch.cat(rgbs, dim=0),              # [T, 3, H, W]
            "gt": torch.cat(depths, dim=0),             # [T, 1, H, W] dense GT
            "dep": torch.cat(lidars, dim=0),            # [T, 1, H, W] sparse LiDAR
            "intrinsic": torch.cat(intrinsics, dim=0),  # [T, 3, 3]
            "scene_id": os.path.basename(scene_dir),
            "idx": idx,
        }
