import random

import numpy as np
import torch
from torch.utils.data import Dataset


def clip_seed(base_seed, idx):
    """Seed of clip idx. Depends only on (base_seed, idx), never on what ran before."""
    return (int(base_seed) * 1000003 + int(idx)) % (2 ** 32)


class SeededClips(Dataset):
    """Reseed python / numpy / torch before loading each clip.

    The video loaders draw random sparse anchors from the global RNGs, so without this
    the anchors would change from run to run. With it, clip idx always gets the same
    anchors for a given base seed (--random-seed); pass different base seeds for
    multi-seed reporting.
    """

    def __init__(self, dataset, base_seed):
        self.dataset = dataset
        self.base_seed = int(base_seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        s = clip_seed(self.base_seed, idx)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        return self.dataset[idx]
