"""
方案四：混合划分（域偏移 + 类别异构）

同时模拟域偏移和类别异构两种 Non-IID 场景。
先将数据按来源分组（合成 gcd vs 真实 ugcd_full），再在来源内按类别进一步划分。
负样本 (ucd/ugcd) 可作为独立客户端或合并到真实数据组。

用法:
    python partition_hybrid.py
    python partition_hybrid.py --domain_clients 2,2 --classes_per_client 2
    python partition_hybrid.py --dry_run
"""

import argparse
import random
from collections import defaultdict

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
    split_list,
)


def partition_hybrid(
    data_root: str,
    domain_clients: list[int],
    classes_per_client: int = 1,
    neg_strategy: str = "separate",
    seed: int = 42,
) -> dict:
    """混合划分：域偏移 + 类别异构。

    Args:
        data_root: 数据集根目录
        domain_clients: 每个域的客户端数
            [0]: gcd（合成数据）域的客户端数
            [1]: ugcd_full（真实数据）域的客户端数
        classes_per_client: 每客户端持有的类别数
        neg_strategy: 负样本策略 ("separate" / "merge_real" / "exclude")
        seed: 随机种子
    """
    rng = random.Random(seed)
    assert len(domain_clients) == 2, "domain_clients 需要指定 [gcd客户端数, ugcd_full客户端数]"

    print("正在扫描数据集...")
    source_data = scan_all_sources(data_root)
    print()

    # ─── 1. 按域收集带类别样本 ───

    # 域 A：合成数据 (gcd)
    domain_a_samples = source_data.get("gcd", [])
    domain_a_classes = get_sample_classes(domain_a_samples)

    # 域 B：真实数据 (ugcd_full，无类别标签)
    domain_b_samples = source_data.get("ugcd_full", [])

    # 负样本
    neg_samples = []
    for src in ["ucd", "ugcd"]:
        neg_samples.extend(source_data.get(src, []))

    print(f"域 A (gcd 合成数据): {len(domain_a_samples)} 样本")
    for cls_id in sorted(domain_a_classes.keys()):
        name = CLASS_NAMES.get(cls_id, str(cls_id))
        print(f"  类别 {cls_id} ({name}): {len(domain_a_classes[cls_id])} 样本")
    print(f"域 B (ugcd_full 真实数据): {len(domain_b_samples)} 样本")
    print(f"负样本 (ucd+ugcd): {len(neg_samples)} 样本")
    print()

    # ─── 2. 域 A：按类别划分 ───
    client_data = {}
    client_idx = 0
    client_domain_map = {}

    available_classes = sorted(domain_a_classes.keys())
    num_classes = len(available_classes)

    if classes_per_client >= num_classes:
        # 每个客户端拥有所有类别
        for c in range(domain_clients[0]):
            cid = f"client_{client_idx}"
            client_data[cid] = list(domain_a_samples)
            rng.shuffle(client_data[cid])
            client_domain_map[cid] = "gcd"
            client_idx += 1
    else:
        # 将类别分配给客户端
        shuffled = available_classes[:]
        rng.shuffle(shuffled)
        num_groups = (num_classes + classes_per_client - 1) // classes_per_client

        # 如果客户端数 > 类别组数，复用类别分配
        for c in range(domain_clients[0]):
            group_idx = c % num_groups
            start = group_idx * classes_per_client
            end = min(start + classes_per_client, num_classes)
            group_classes = shuffled[start:end]

            samples = []
            for cls_id in group_classes:
                samples.extend(domain_a_classes.get(cls_id, []))
            rng.shuffle(samples)

            cid = f"client_{client_idx}"
            client_data[cid] = samples
            client_domain_map[cid] = "gcd"
            client_idx += 1

    # ─── 3. 域 B：均匀或按伪标签划分 ───
    # ugcd_full 没有明确的类别标签，均匀分配
    if domain_b_samples:
        b_splits = split_list(domain_b_samples, domain_clients[1], seed=seed + 50)
        for i, split in enumerate(b_splits):
            cid = f"client_{client_idx}"
            client_data[cid] = split
            client_domain_map[cid] = "ugcd_full"
            client_idx += 1

    # ─── 4. 负样本处理 ───
    if neg_strategy == "separate" and neg_samples:
        cid = f"client_{client_idx}"
        client_data[cid] = neg_samples
        client_domain_map[cid] = "neg"
        client_idx += 1
    elif neg_strategy == "merge_real" and neg_samples:
        # 将负样本均匀分配到域 B 客户端
        real_cids = [c for c, d in client_domain_map.items() if d == "ugcd_full"]
        if real_cids:
            rng.shuffle(neg_samples)
            for i, s in enumerate(neg_samples):
                client_data[real_cids[i % len(real_cids)]].append(s)
        else:
            # 没有真实数据客户端，单独创建
            cid = f"client_{client_idx}"
            client_data[cid] = neg_samples
            client_domain_map[cid] = "neg"
            client_idx += 1

    return build_partition_output(
        method="hybrid",
        params={
            "domain_clients": {
                "gcd": domain_clients[0],
                "ugcd_full": domain_clients[1],
            },
            "classes_per_client": classes_per_client,
            "neg_strategy": neg_strategy,
            "client_domain_map": client_domain_map,
        },
        client_data=client_data,
    )


def main():
    parser = argparse.ArgumentParser(description="方案四：混合划分（域偏移 + 类别异构）")
    add_common_args(parser)
    parser.add_argument(
        "--domain_clients",
        type=str,
        default="3,2",
        help="每域客户端数，逗号分隔: gcd客户端数,ugcd_full客户端数 (默认: 3,2)",
    )
    parser.add_argument(
        "--classes_per_client",
        type=int,
        default=1,
        help="gcd 域内每客户端持有的类别数 (1=极端, 2=中等)",
    )
    parser.add_argument(
        "--neg_strategy",
        type=str,
        default="separate",
        choices=["separate", "merge_real", "exclude"],
        help="负样本策略: separate=单独客户端, merge_real=合并到真实数据客户端, exclude=排除",
    )
    args = parser.parse_args()

    dc = [int(x) for x in args.domain_clients.split(",")]
    result = partition_hybrid(
        args.data_root,
        domain_clients=dc,
        classes_per_client=args.classes_per_client,
        neg_strategy=args.neg_strategy,
        seed=args.seed,
    )

    print_partition_summary(result)

    if not args.dry_run:
        output_path = (
            f"{args.output_dir}/partition_hybrid"
            f"_c{args.classes_per_client}_{args.neg_strategy}.json"
        )
        save_partition(result, output_path)


if __name__ == "__main__":
    main()
