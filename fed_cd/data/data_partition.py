"""
Partition loader for federated change detection.

Loads a partition JSON file and provides:
  - client sample lists (for federated training)
  - centralized train/val/test splits (for baseline)
  - evaluation set splits
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import Dataset


def load_partition(partition_path: str) -> dict:
    """Load a partition JSON file.

    Returns:
        dict with keys: partition_method, num_clients, params, clients
        Each client has: samples (list of dicts), stats
    """
    with open(partition_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_client_samples(partition_path: str, client_id: str) -> list[dict]:
    """Get the sample list for a specific client."""
    partition = load_partition(partition_path)
    return partition["clients"][client_id]["samples"]


def get_all_client_ids(partition_path: str) -> list[str]:
    """Get sorted list of client IDs."""
    partition = load_partition(partition_path)
    return sorted(partition["clients"].keys(), key=lambda x: int(x.split("_")[1]))


def scan_evaluation_set(data_root: str, split: str) -> list[dict]:
    """Scan val/test/test2 split for evaluation. Stores relative paths.

    Args:
        data_root: path to WHU-GCD/ directory
        split: 'val', 'test', or 'test2'

    Returns:
        list of sample dicts with keys: im1, im2, label, mask1, mask2, source
        Paths are relative to data_root (e.g. 'val/im1/E10_63.png')
    """
    split_dir = Path(data_root) / split
    im2_dir = split_dir / "im2"
    if not im2_dir.exists():
        print(f"Warning: {im2_dir} does not exist")
        return []

    def _resolve_im1_name(im2_name):
        """Resolve im1 filename from im2 filename.

        For synthetic sets (val/test): im2='E10_20_6.png' -> im1='E10_20.png'
        For real-world sets (test2): im2='clcd_0.png' -> im1='clcd_0.png' (same)
        """
        im1_dir = split_dir / "im1"
        direct = im1_dir / im2_name
        if direct.exists():
            return im2_name

        stem = Path(im2_name).stem
        parts = stem.split("_")
        if len(parts) >= 2:
            try:
                int(parts[-1])
                im1_stem = "_".join(parts[:-1])
                candidate = im1_stem + ".png"
                if (im1_dir / candidate).exists():
                    return candidate
            except ValueError:
                pass
        return None

    samples = []
    for im2_file in sorted(im2_dir.glob("*.png")):
        im2_name = im2_file.name
        im1_name = _resolve_im1_name(im2_name)
        if im1_name is None:
            continue

        label_file = split_dir / "label" / im2_name
        if not label_file.is_file():
            continue

        mask1_dir_exists = (split_dir / "mask1").exists()
        mask2_dir_exists = (split_dir / "mask2").exists()
        mask1_file = split_dir / "mask1" / im1_name if mask1_dir_exists else None
        mask2_file = split_dir / "mask2" / im2_name if mask2_dir_exists else None

        samples.append({
            "im1": f"{split}/im1/{im1_name}",
            "im2": f"{split}/im2/{im2_name}",
            "label": f"{split}/label/{im2_name}",
            "mask1": f"{split}/mask1/{im1_name}" if (mask1_file and mask1_file.is_file()) else "",
            "mask2": f"{split}/mask2/{im2_name}" if (mask2_file and mask2_file.is_file()) else "",
            "source": split,
        })

    print(f"  {split}: {len(samples)} samples")
    return samples


def collect_centralized_train(data_root: str, sources: Optional[list[str]] = None) -> list[dict]:
    """Collect all training samples from specified sources for centralized baseline.

    Args:
        data_root: path to WHU-GCD/ directory
        sources: list of source names to include. Default: all 4 sources.

    Returns:
        list of sample dicts
    """
    if sources is None:
        sources = ["gcd", "ugcd_full", "ucd", "ugcd"]

    from partition_utils import scan_all_sources

    all_data = scan_all_sources(data_root)
    samples = []
    for src in sources:
        samples.extend(all_data.get(src, []))
    return samples
