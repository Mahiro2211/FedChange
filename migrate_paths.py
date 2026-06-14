#!/usr/bin/env python
"""
Migrate partition JSON files from absolute paths to relative paths.

This script converts all sample paths in partition JSONs from machine-specific
absolute paths (e.g. 'C:\\Dataset\\WHU-GCD\\WHU-GCD\\train\\gcd\\...') to
portable relative paths (e.g. 'train/gcd/im1/...').

After migration, the JSONs work on ANY machine — just set --data_root to
point to the WHU-GCD dataset directory.

Usage:
    python migrate_paths.py                    # migrate all JSONs in partitions/
    python migrate_paths.py --check            # dry-run, only show what would change
    python migrate_paths.py --file partitions/partition_source.json
"""

import argparse
import json
import os
import re
from pathlib import Path


SPLIT_PREFIXES = ["train/", "val/", "test/", "test2/"]


def to_relative(path: str) -> str:
    """Convert an absolute path to relative (if possible).

    Looks for known split prefixes (train/, val/, test/, test2/) and
    extracts everything from that point onward.

    Examples:
        'C:\\Dataset\\WHU-GCD\\WHU-GCD\\train\\gcd\\im1\\x.png'
            -> 'train/gcd/im1/x.png'
        '/home/user/data/WHU-GCD/val/im2/y.png'
            -> 'val/im2/y.png'
        'train/gcd/im1/x.png' (already relative)
            -> 'train/gcd/im1/x.png' (unchanged)
    """
    if not path:
        return path

    # Normalize separators to forward slash
    normalized = path.replace("\\", "/")

    # Check if already relative (doesn't start with drive letter or /)
    if not os.path.isabs(normalized.replace("/", os.sep)) and not normalized.startswith(("/", *SPLIT_PREFIXES)):
        # Could already be relative, check if it starts with a known prefix
        for prefix in SPLIT_PREFIXES:
            if normalized.startswith(prefix):
                return normalized
        # Otherwise might be a weird relative path, leave as-is
        return normalized

    # Find the split prefix in the path
    for prefix in SPLIT_PREFIXES:
        idx = normalized.find("/" + prefix)
        if idx != -1:
            return normalized[idx + 1:]  # +1 to skip the leading /

    # Also check without leading slash (for paths already starting with prefix)
    for prefix in SPLIT_PREFIXES:
        if normalized.startswith(prefix):
            return normalized

    # Could not find a known prefix — leave as-is
    return normalized


def migrate_file(json_path: str, check_only: bool = False) -> tuple[int, int]:
    """Migrate a single partition JSON file.

    Returns:
        (num_changed, num_total) — number of paths changed and total paths
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_changed = 0
    num_total = 0
    path_keys = ["im1", "im2", "label", "mask1", "mask2"]

    for cid, cdata in data.get("clients", {}).items():
        for sample in cdata.get("samples", []):
            for key in path_keys:
                old_val = sample.get(key, "")
                if not old_val:
                    continue
                num_total += 1
                new_val = to_relative(old_val)
                if new_val != old_val:
                    num_changed += 1
                    sample[key] = new_val

    if num_changed > 0 and not check_only:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return num_changed, num_total


def main():
    parser = argparse.ArgumentParser(description="Migrate partition JSONs to relative paths")
    parser.add_argument("--partitions_dir", type=str, default="partitions",
                        help="directory containing partition JSON files")
    parser.add_argument("--file", type=str, default=None,
                        help="migrate a specific file (default: all JSONs in partitions_dir)")
    parser.add_argument("--check", action="store_true",
                        help="dry-run: only show what would change, don't write")
    args = parser.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = sorted(Path(args.partitions_dir).glob("*.json"))

    if not files:
        print(f"No JSON files found in {args.partitions_dir}/")
        return

    mode = "CHECK" if args.check else "MIGRATE"
    print(f"\n{'='*60}")
    print(f"  Path Migration ({mode})")
    print(f"{'='*60}")

    total_changed = 0
    total_paths = 0

    for json_path in files:
        changed, total = migrate_file(str(json_path), check_only=args.check)
        total_changed += changed
        total_paths += total
        status = "would change" if args.check else "changed"
        if changed > 0:
            print(f"  {json_path.name}: {changed}/{total} paths {status}")
        else:
            print(f"  {json_path.name}: already relative ({total} paths OK)")

    print(f"\n  Total: {total_changed}/{total_paths} paths {'would be ' if args.check else ''}changed")
    if not args.check and total_changed > 0:
        print(f"  All partition JSONs now use portable relative paths.")
    print(f"  Set --data_root to your WHU-GCD directory at runtime.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
