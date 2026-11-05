"""DDAD validation split, video clips.

Layout (all scenes in one folder, files sorted scene after scene):

    <dataset_path>/rgb/<frame>.png      RGB, 1216x1936
    <dataset_path>/gt/<frame>.png       dense GT, uint16 / 256 = metres   (any '*gt' folder)
    <dataset_path>/hints/<frame>.png    sparse hints, uint16 / 256 = metres

The sorted lists hold the scenes back to back in the key order of
ddad_valset_info.json (scene -> frame count), so scene k starts at the sum of the
frame counts before it. Clips are T consecutive frames of one scene, non-overlapping.
Depths are clipped to 80 m.
"""

import json
import os
import os.path as osp
from glob import glob

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

with open(osp.join(osp.dirname(osp.abspath(__file__)), 'ddad_valset_info.json')) as _f:
    valset_num_frames = json.load(_f)


def read_depth(file_name, max_depth=None):
    """uint16 PNG / 256 -> metres, optionally clipped to max_depth."""
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_depth = np.array(Image.open(file_name))
    assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
        "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)
    image_depth = image_depth.astype(np.float32) / 256.0
    if max_depth is not None:
        image_depth = np.clip(image_depth, 0, max_depth)
    return image_depth


class valset(Dataset):
    def __init__(self, args):
        self.T = int(args.num_frames)
        assert self.T > 0, "num_frames must be > 0"
        if self.T > 50:
            raise ValueError("num_frames cannot exceed the scene length (50).")

        datapath = os.path.expanduser(args.dataset_path)
        image_list = sorted(glob(osp.join(datapath, 'rgb/*.png')))
        gt_list = sorted(glob(osp.join(datapath, '*gt/*.png')))
        hints_list = sorted(glob(osp.join(datapath, 'hints/*.png')))
        n = len(image_list)
        assert n > 0 and len(gt_list) == n and len(hints_list) == n, \
            f"rgb/gt/hints counts must match and be > 0 in {datapath}"
        assert sum(valset_num_frames.values()) == n, \
            "ddad_valset_info.json frame counts must add up to the number of rgb files"

        scenes = {}
        offset = 0
        for scene_id, count in valset_num_frames.items():
            scenes[scene_id] = [{"rgb": image_list[i], "gt": gt_list[i], "depth": hints_list[i]}
                                for i in range(offset, offset + count)]
            offset += count

        self.clips = []                     # (scene_id, start frame), stride T, tail dropped
        for sid in sorted(scenes.keys()):
            for k in range(len(scenes[sid]) // self.T):
                self.clips.append((sid, k * self.T))
        self._scenes = scenes

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        sid, start = self.clips[idx]
        rgbs, deps, gts = [], [], []
        for frame in self._scenes[sid][start:start + self.T]:
            rgb = Image.open(frame["rgb"])
            depth = Image.fromarray(read_depth(frame["depth"], max_depth=80.0).astype('float32'), mode='F')
            gt = Image.fromarray(read_depth(frame["gt"], max_depth=80.0).astype('float32'), mode='F')
            assert rgb.size == depth.size == gt.size

            rgb = TF.normalize(TF.to_tensor(rgb), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            rgbs.append(rgb)
            deps.append(TF.to_tensor(np.array(depth)))
            gts.append(TF.to_tensor(np.array(gt)))

        return {
            "rgb": torch.stack(rgbs, dim=0),    # [T,3,H,W]
            "dep": torch.stack(deps, dim=0),    # [T,1,H,W]
            "gt": torch.stack(gts, dim=0),      # [T,1,H,W]
            "idx": idx,
            "scene_id": sid,
        }
