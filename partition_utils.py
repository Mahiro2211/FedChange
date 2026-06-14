"""
WHU-GCD 联邦学习数据划分 —— 公共工具模块

提供数据集扫描、类别解析、统计输出、JSON 格式化等共用功能，
供 partition_by_source / partition_by_class / partition_dirichlet / partition_hybrid 调用。
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# ──────────────────────── 常量 ────────────────────────

# 语义类别 ID → 中文名（与 WHU-GCD readme 一致）
CLASS_NAMES = {
    2: "建筑",
    3: "道路",
    4: "水体",
    5: "荒地",
    6: "森林",
    7: "农业",
}

# 四种训练数据来源
SOURCES = ["gcd", "ugcd_full", "ucd", "ugcd"]

# 每个来源子目录下标准文件夹
SUBDIRS = ["im1", "im2", "label", "mask1", "mask2"]


# ──────────────────────── 数据集扫描 ────────────────────────

def scan_source(data_root: str, source: str) -> list[dict[str, str]]:
    """扫描某一来源目录，返回样本列表。

    每个样本是一个字典（路径相对于 data_root，便于跨平台迁移）：
        {
            "im1":   "train/gcd/im1/E10_0.png",
            "im2":   "train/gcd/im2/E10_0_5.png",
            "label": "train/gcd/label/E10_0_5.png",
            "mask1": "train/gcd/mask1/E10_0.png",
            "mask2": "train/gcd/mask2/E10_0_5.png",
            "source": "gcd",
        }

    对于 gcd，im2 文件名含类别后缀 (x_i.png)；其余来源不含。
    """
    src_dir = Path(data_root) / "train" / source
    im2_dir = src_dir / "im2"
    if not im2_dir.exists():
        return []

    samples = []
    for im2_file in sorted(im2_dir.glob("*.png")):
        im2_name = im2_file.name  # e.g. "E10_0_5.png"

        # 构造 im1 文件名：gcd 需要去掉最后的类别后缀，其余来源直接同名
        im1_name = _im2_to_im1_name(im2_name, source)

        # 构造相对路径（相对于 data_root）
        rel_prefix = f"train/{source}"
        im1_rel = f"{rel_prefix}/im1/{im1_name}"
        im2_rel = f"{rel_prefix}/im2/{im2_name}"
        label_rel = f"{rel_prefix}/label/{im2_name}"
        mask1_rel = f"{rel_prefix}/mask1/{im1_name}"
        mask2_rel = f"{rel_prefix}/mask2/{im2_name}"

        # 检查关键文件是否存在（im1 和 label 必须存在）
        im1_path = str(Path(data_root) / im1_rel)
        label_path = str(Path(data_root) / label_rel)
        if not os.path.isfile(im1_path):
            continue
        if not os.path.isfile(label_path):
            continue

        mask1_path = str(Path(data_root) / mask1_rel)
        mask2_path = str(Path(data_root) / mask2_rel)

        samples.append({
            "im1": im1_rel,
            "im2": im2_rel,
            "label": label_rel,
            "mask1": mask1_rel if os.path.isfile(mask1_path) else "",
            "mask2": mask2_rel if os.path.isfile(mask2_path) else "",
            "source": source,
        })

    return samples


def scan_all_sources(data_root: str) -> dict[str, list[dict[str, str]]]:
    """扫描所有训练数据来源，返回 {来源名: [样本列表]}。"""
    result = {}
    for source in SOURCES:
        samples = scan_source(data_root, source)
        result[source] = samples
        print(f"  {source:12s}: {len(samples):>6d} 样本")
    return result


def scan_evaluation(data_root: str, split: str) -> list[dict[str, str]]:
    """扫描验证集或测试集。路径相对于 data_root 存储。

    Args:
        split: "val" / "test" / "test2"
    """
    split_dir = Path(data_root) / split
    im2_dir = split_dir / "im2"
    if not im2_dir.exists():
        return []

    samples = []
    for im2_file in sorted(im2_dir.glob("*.png")):
        im2_name = im2_file.name
        # 验证集/测试集的 im1 与 im2 同名（或需要去后缀）
        im1_name = im2_name

        rel_prefix = split
        im1_rel = f"{rel_prefix}/im1/{im1_name}"
        im2_rel = f"{rel_prefix}/im2/{im2_name}"
        label_rel = f"{rel_prefix}/label/{im2_name}"

        im1_path = str(Path(data_root) / im1_rel)
        label_path = str(Path(data_root) / label_rel)

        if not os.path.isfile(im1_path) or not os.path.isfile(label_path):
            continue

        mask1_path = str(Path(data_root) / split / "mask1" / im1_name) if (split_dir / "mask1").exists() else ""
        mask2_path = str(Path(data_root) / split / "mask2" / im2_name) if (split_dir / "mask2").exists() else ""

        mask1_rel = f"{rel_prefix}/mask1/{im1_name}" if (split_dir / "mask1").exists() else ""
        mask2_rel = f"{rel_prefix}/mask2/{im2_name}" if (split_dir / "mask2").exists() else ""

        samples.append({
            "im1": im1_rel,
            "im2": im2_rel,
            "label": label_rel,
            "mask1": mask1_rel if os.path.isfile(mask1_path) else "",
            "mask2": mask2_rel if os.path.isfile(mask2_path) else "",
            "source": split,
        })

    return samples


# ──────────────────────── 类别解析 ────────────────────────

def parse_change_class(im2_filename: str, source: str) -> int | None:
    """从 im2 文件名中解析变化类别 ID。

    gcd 样例: "E10_0_5.png" → 5
    其余来源: 无类别后缀 → None
    """
    if source != "gcd":
        return None
    stem = Path(im2_filename).stem  # "E10_0_5"
    parts = stem.split("_")
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            return None
    return None


def get_sample_classes(samples: list[dict]) -> dict[int, list[dict]]:
    """将样本按变化类别分组。仅对 gcd 来源有效。"""
    class_groups: dict[int, list[dict]] = defaultdict(list)
    for s in samples:
        cls = parse_change_class(Path(s["im2"]).name, s["source"])
        if cls is not None:
            class_groups[cls].append(s)
    return dict(class_groups)


# ──────────────────────── 划分辅助 ────────────────────────

def split_list(lst: list, n: int, seed: int = 42) -> list[list]:
    """将列表尽量均匀地分成 n 份。"""
    rng = random.Random(seed)
    shuffled = lst[:]
    rng.shuffle(shuffled)
    k, m = divmod(len(shuffled), n)
    return [shuffled[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def dirichlet_partition(labels: list[int], num_clients: int, alpha: float,
                        seed: int = 42) -> list[list[int]]:
    """Dirichlet 非独立同分布划分。

    Args:
        labels: 每个样本的类别标签列表
        num_clients: 客户端数量
        alpha: Dirichlet 浓度参数（越小越异构）
        seed: 随机种子

    Returns:
        每个客户端分到的样本索引列表
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    unique_labels = sorted(set(labels))

    # 每个类别的样本索引
    label_indices: dict[int, list[int]] = defaultdict(list)
    for idx, lbl in enumerate(labels):
        label_indices[lbl].append(idx)

    # 每个类别在客户端间的分配比例
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for lbl in unique_labels:
        idxs = label_indices[lbl]
        proportions = rng.dirichlet([alpha] * num_clients)
        # 按比例分配
        proportions = (proportions * len(idxs)).astype(int)
        # 修正四舍五入误差
        diff = len(idxs) - proportions.sum()
        for i in range(abs(diff)):
            proportions[i % num_clients] += 1 if diff > 0 else -1

        offset = 0
        for c in range(num_clients):
            client_indices[c].extend(idxs[offset:offset + proportions[c]])
            offset += proportions[c]

    return client_indices


# ──────────────────────── 统计与输出 ────────────────────────

def compute_client_stats(client_id: str, samples: list[dict]) -> dict[str, Any]:
    """计算单个客户端的数据统计信息。"""
    stats: dict[str, Any] = {
        "total": len(samples),
        "sources": defaultdict(int),
        "class_dist": defaultdict(int),
    }
    for s in samples:
        stats["sources"][s["source"]] += 1
        cls = parse_change_class(Path(s["im2"]).name, s["source"])
        if cls is not None:
            stats["class_dist"][str(cls)] += 1
        else:
            stats["class_dist"]["unknown"] += 1

    # 转换为普通 dict 以便 JSON 序列化
    stats["sources"] = dict(stats["sources"])
    stats["class_dist"] = dict(stats["class_dist"])
    return stats


def build_partition_output(method: str, params: dict,
                           client_data: dict[str, list[dict]]) -> dict:
    """构建统一的划分结果字典。"""
    clients = {}
    for cid, samples in client_data.items():
        clients[cid] = {
            "samples": samples,
            "stats": compute_client_stats(cid, samples),
        }
    return {
        "partition_method": method,
        "num_clients": len(client_data),
        "params": params,
        "clients": clients,
    }


def save_partition(result: dict, output_path: str) -> None:
    """将划分结果保存为 JSON 文件。"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"划分结果已保存: {output_path}")


def print_partition_summary(result: dict) -> None:
    """打印划分结果的可读摘要。"""
    print(f"\n{'=' * 60}")
    print(f"划分方式: {result['partition_method']}")
    print(f"参数: {result['params']}")
    print(f"客户端数: {result['num_clients']}")
    print(f"{'=' * 60}")

    total_samples = 0
    for cid, cdata in result["clients"].items():
        stats = cdata["stats"]
        total = stats["total"]
        total_samples += total

        # 来源分布
        src_str = ", ".join(f"{k}={v}" for k, v in stats["sources"].items())

        # 类别分布
        cls_parts = []
        for cls_id, cnt in sorted(stats["class_dist"].items()):
            name = CLASS_NAMES.get(int(cls_id), cls_id) if cls_id != "unknown" else "未知"
            cls_parts.append(f"{name}={cnt}")
        cls_str = ", ".join(cls_parts)

        print(f"\n  {cid} ({total} 样本)")
        print(f"    来源: {src_str}")
        print(f"    类别: {cls_str}")

    print(f"\n总样本数: {total_samples}")
    print(f"{'=' * 60}")


def print_partition_summary_grouped(result: dict) -> None:
    """按主类别分组的客户端摘要（适用于大客户端数场景，如 FedSeg 风格）。

    将客户端按其主要变化类别分组，每组显示客户端数、样本数 min/mean/max，
    避免在客户端数 ≥ 60 时逐客户端打印导致日志过长。
    """
    print(f"\n{'=' * 60}")
    print(f"划分方式: {result['partition_method']}")
    print(f"参数: {result['params']}")
    print(f"客户端数: {result['num_clients']}")
    print(f"{'=' * 60}")

    # 按主类别分组：每客户端只看 class_dist 的唯一键
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    total_samples = 0
    for cid, cdata in result["clients"].items():
        stats = cdata["stats"]
        total = stats["total"]
        total_samples += total

        class_dist = stats["class_dist"]
        if not class_dist:
            group_key = "empty"
        elif "unknown" in class_dist and len(class_dist) == 1:
            group_key = "neg"  # 负样本（无类别后缀）
        else:
            # 取数量最多的类作为主类
            group_key = max(class_dist.items(), key=lambda x: x[1])[0]
        groups[group_key].append((cid, total))

    # 排序：负样本排最后，其他按类别 ID 升序
    def _sort_key(k: str) -> tuple:
        if k in ("neg", "empty"):
            return (1, k)
        try:
            return (0, int(k))
        except ValueError:
            return (1, k)

    print(f"\n按主类别分组:")
    for group_key in sorted(groups.keys(), key=_sort_key):
        clients = groups[group_key]
        sizes = [s for _, s in clients]
        if group_key == "neg":
            name = "负样本 (ucd+ugcd+ugcd_full)"
        elif group_key == "empty":
            name = "空客户端"
        else:
            name = f"类别 {group_key} ({CLASS_NAMES.get(int(group_key), '?')})"
        print(f"\n  {name}  [{len(clients)} 个客户端]")
        print(f"    样本数: min={min(sizes)}, mean={sum(sizes) / len(sizes):.0f}, "
              f"max={max(sizes)}, 总计={sum(sizes)}")
        # 列出前 3 个和后 1 个客户端 ID，便于抽查
        if len(clients) <= 4:
            preview = ", ".join(f"{c}={s}" for c, s in clients)
        else:
            head = ", ".join(f"{c}={s}" for c, s in clients[:3])
            tail = f"{clients[-1][0]}={clients[-1][1]}"
            preview = f"{head}, ..., {tail}"
        print(f"    客户端: {preview}")

    print(f"\n总样本数: {total_samples}")
    print(f"{'=' * 60}")


# ──────────────────────── 内部工具 ────────────────────────

def _im2_to_im1_name(im2_name: str, source: str) -> str:
    """将 im2 文件名转换为对应的 im1 文件名。

    gcd: "E10_0_5.png" → "E10_0.png"（去掉最后的类别后缀）
    其余: "Ub1_0.png" → "Ub1_0.png"（直接同名）
    """
    if source != "gcd":
        return im2_name
    stem = Path(im2_name).stem  # "E10_0_5"
    parts = stem.split("_")
    # 尝试去掉最后一个数值后缀
    if len(parts) >= 2:
        try:
            int(parts[-1])
            im1_stem = "_".join(parts[:-1])
            return im1_stem + ".png"
        except ValueError:
            return im2_name
    return im2_name


# ──────────────────────── CLI 公共参数 ────────────────────────

def add_common_args(parser):
    """为子脚本的 argparse 添加公共参数。"""
    parser.add_argument(
        "--data_root",
        type=str,
        default=r"..\WHU-GCD",
        help="WHU-GCD 数据集根目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./partitions",
        help="划分结果 JSON 输出目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅打印统计信息，不写入文件",
    )
