"""
WHU-GCD 联邦学习 Dirichlet Non-IID 数据划分。

使用 Dirichlet 分布控制不同客户端之间的类别分布异构程度。
alpha 越小异构性越强，alpha→∞ 等价于 IID 均匀划分。
    0.05 ~ 0.1: 极端异构
    0.5:        中等异构
    1.0:        轻度异构
    ≥10:        接近 IID

本模块同时是：(1) 被其他代码 import 的工具库（数据扫描 scan_all_sources、
划分 dirichlet_partition、统计/保存等）；(2) 生成划分 JSON 的 CLI 入口。

用法（从仓库根目录 FedChange/ 运行）:
    # 默认参数生成
    python -m partitions.generate --data_root ../WHU-GCD

    # 自定义 alpha / 客户端数
    python -m partitions.generate --alpha 0.1 --num_clients 10 --data_root ../WHU-GCD

    # 仅预览统计，不写文件
    python -m partitions.generate --alpha 1.0 --dry_run
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# ──────────────────────── 常量 ────────────────────────

# 语义类别 ID → 中文名（与 WHU-GCD readme 一致）
# 0 = 无变化（负样本 ucd/ugcd/ugcd_full，作为独立的语义类参与 Non-IID 划分）
CLASS_NAMES = {
    0: "无变化",
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
            # 无类别后缀的样本（ucd/ugcd/ugcd_full 负样本）记为类 0（无变化）
            stats["class_dist"]["0"] += 1

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
        for cls_id, cnt in sorted(stats["class_dist"].items(), key=lambda x: int(x[0]) if x[0].isdigit() else 99):
            name = CLASS_NAMES.get(int(cls_id), cls_id)
            cls_parts.append(f"{name}={cnt}")
        cls_str = ", ".join(cls_parts)

        print(f"\n  {cid} ({total} 样本)")
        print(f"    来源: {src_str}")
        print(f"    类别: {cls_str}")

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


# ──────────────────────── Dirichlet 划分 ────────────────────────

def partition_dirichlet(
    data_root: str,
    num_clients: int,
    alpha: float,
    include_neg: bool = True,
    seed: int = 42,
) -> dict:
    """Dirichlet 非独立同分布划分。

    Args:
        data_root: 数据集根目录
        num_clients: 客户端数量
        alpha: Dirichlet 浓度参数
            0.05 ~ 0.1: 极端异构
            0.5: 中等异构
            1.0: 轻度异构
            ≥10: 接近 IID
        include_neg: 是否包含负样本 (ucd/ugcd/ugcd_full)
        seed: 随机种子
    """
    print("正在扫描数据集...")
    source_data = scan_all_sources(data_root)
    print()

    # ─── 1. 收集所有样本并标注类别 ───
    all_samples = []
    sample_labels = []

    # gcd 样本：有明确的类别标签 (2-7)
    gcd_samples = source_data.get("gcd", [])
    class_groups = get_sample_classes(gcd_samples)

    print("gcd 类别分布:")
    for cls_id in sorted(class_groups.keys()):
        name = CLASS_NAMES.get(cls_id, str(cls_id))
        print(f"  类别 {cls_id} ({name}): {len(class_groups[cls_id])} 样本")
    print()

    for cls_id, samples in class_groups.items():
        for s in samples:
            all_samples.append(s)
            sample_labels.append(cls_id)

    # 负样本：标记为类别 0（无变化）
    if include_neg:
        for src in ["ucd", "ugcd", "ugcd_full"]:
            for s in source_data.get(src, []):
                all_samples.append(s)
                sample_labels.append(0)

    num_with_class = len([l for l in sample_labels if l > 0])
    num_neg = len([l for l in sample_labels if l == 0])
    print(f"有变化样本: {num_with_class}, 负样本: {num_neg}, 总计: {len(all_samples)}")

    # ─── 2. Dirichlet 划分 ───
    print(f"\n执行 Dirichlet 划分: alpha={alpha}, num_clients={num_clients}")
    client_indices = dirichlet_partition(sample_labels, num_clients, alpha, seed=seed)

    # ─── 3. 构建客户端数据 ───
    client_data = {}
    for c in range(num_clients):
        cid = f"client_{c}"
        indices = client_indices[c]
        client_data[cid] = [all_samples[i] for i in indices]

    return build_partition_output(
        method="dirichlet",
        params={
            "alpha": alpha,
            "num_clients": num_clients,
            "include_neg": include_neg,
        },
        client_data=client_data,
    )


# ──────────────────────── CLI ────────────────────────

def add_common_args(parser):
    """为 CLI 添加公共参数。"""
    parser.add_argument(
        "--data_root",
        type=str,
        default=r"../WHU-GCD",
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


def main():
    parser = argparse.ArgumentParser(description="Dirichlet 分布划分（可控异构度）")
    add_common_args(parser)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet 浓度参数 (0.1=极端异构, 0.5=中等, 1.0=轻度, 10=接近IID)",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=7,
        help="客户端数量",
    )
    parser.add_argument(
        "--no_neg",
        action="store_true",
        help="排除负样本 (ucd/ugcd/ugcd_full)",
    )
    args = parser.parse_args()

    result = partition_dirichlet(
        args.data_root,
        num_clients=args.num_clients,
        alpha=args.alpha,
        include_neg=not args.no_neg,
        seed=args.seed,
    )

    print_partition_summary(result)

    if not args.dry_run:
        output_path = (
            f"{args.output_dir}/partition_dirichlet_a{args.alpha}"
            f"_n{args.num_clients}.json"
        )
        save_partition(result, output_path)


if __name__ == "__main__":
    main()
