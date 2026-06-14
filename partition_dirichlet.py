"""
方案三：Dirichlet 分布划分（通用可控异构度）

使用 Dirichlet 分布控制不同客户端之间的类别分布异构程度。
alpha 越小异构性越强，alpha→∞ 等价于 IID 均匀划分。

用法:
    python partition_dirichlet.py --alpha 0.5
    python partition_dirichlet.py --alpha 0.1 --num_clients 10
    python partition_dirichlet.py --alpha 1.0 --dry_run
"""

import argparse
import random
from pathlib import Path

from partition_utils import (
    CLASS_NAMES,
    SOURCES,
    add_common_args,
    build_partition_output,
    dirichlet_partition,
    get_sample_classes,
    parse_change_class,
    print_partition_summary,
    save_partition,
    scan_all_sources,
)


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
        include_neg: 是否包含负样本 (ucd/ugcd)
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


def main():
    parser = argparse.ArgumentParser(description="方案三：Dirichlet 分布划分")
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
