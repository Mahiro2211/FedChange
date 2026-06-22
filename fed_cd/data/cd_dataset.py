"""
WHU-GCD Change Detection Dataset.

Supports both binary change detection (BCD) and semantic change detection (SCD).
Works with partition JSON files for federated learning.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .data_utils import CDDataAugmentation


class CDDataset(Dataset):
    """Change detection dataset for WHU-GCD.

    Each sample is a dict with keys: im1, im2, label, mask1, mask2, source.
    Paths can be relative (to data_root) or absolute (legacy compatibility).
    Returns (im1_tensor, im2_tensor, label_tensor) for BCD.
    Returns (im1_tensor, im2_tensor, label_tensor, mask1_tensor, mask2_tensor) for SCD.

    Args:
        samples: list of sample dicts (from partition JSON or scan functions)
        img_size: target image size (default 256)
        is_train: whether to apply training augmentation
        task: 'bcd' for binary change detection, 'scd' for semantic change detection
        data_root: dataset root for resolving relative paths (default '../WHU-GCD')
    """

    def __init__(self, samples: list[dict], img_size=256, is_train=True,
                 task="bcd", data_root="../WHU-GCD"):
        self.samples = samples
        self.img_size = img_size
        self.is_train = is_train
        self.task = task
        self.data_root = data_root

        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
            )
        else:
            self.augm = CDDataAugmentation(img_size=self.img_size)

    def _resolve_path(self, path: str) -> str:
        """Resolve a path that may be relative or absolute.

        - Absolute paths (e.g. 'C:\\...' or '/home/...') are used as-is.
        - Relative paths (e.g. 'train/gcd/im1/x.png') are joined with data_root.
        """
        if not path:
            return path
        if os.path.isabs(path):
            return path
        return os.path.join(self.data_root, path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        img_a = np.asarray(Image.open(self._resolve_path(sample["im1"])).convert("RGB"))
        img_b = np.asarray(Image.open(self._resolve_path(sample["im2"])).convert("RGB"))
        label = np.array(Image.open(self._resolve_path(sample["label"])), dtype=np.uint8)

        if self.task == "bcd":
            label = label // 255
            imgs, labels = self.augm.transform([img_a, img_b], [label], to_tensor=True)
            return {
                "name": os.path.basename(sample["im2"]),
                "A": imgs[0],
                "B": imgs[1],
                "L": labels[0],
            }
        elif self.task == "scd":
            mask1 = np.array(Image.open(self._resolve_path(sample["mask1"])), dtype=np.uint8) if sample.get("mask1") else np.zeros_like(label)
            mask2 = np.array(Image.open(self._resolve_path(sample["mask2"])), dtype=np.uint8) if sample.get("mask2") else np.zeros_like(label)
            imgs, labels = self.augm.transform([img_a, img_b], [label], to_tensor=True)
            return {
                "name": os.path.basename(sample["im2"]),
                "A": imgs[0],
                "B": imgs[1],
                "L": labels[0],
                "M1": torch.from_numpy(mask1).long(),
                "M2": torch.from_numpy(mask2).long(),
            }
        else:
            raise ValueError(f"Unknown task: {self.task}")


def build_eval_dataset(samples: list[dict], img_size=256, task="bcd", data_root="../WHU-GCD"):
    """Build evaluation dataset from a sample list."""
    return CDDataset(samples, img_size=img_size, is_train=False, task=task, data_root=data_root)
