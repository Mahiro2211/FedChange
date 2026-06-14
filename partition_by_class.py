"""
方案二：按语义类别划分（FedSeg 风格，每类拆分多客户端）

借鉴 FedSeg (CVPR 2023) 的 datasplit.py 划分策略：每个变化类别均匀拆分给
固定数量的客户端（clients_per_class），客户端 ID 遵循 FedSeg 公式
    client_index = i + class_idx × clients_per_class
从而做到"每个客户端只持有单一类别的数据"——极端 Non-IID。

gcd 数据中 im2 文件名含类别后缀 (x_i.png)，i 为变化目标类别 (2-7)。
负样本 (ucd/ugcd/ugcd_full) 通过 --neg_strategy 控制：
    class_like   把负样本视为"类 0"，按 clients_per_class 同样切分（推荐，最贴 FedSeg）
    separate     负样本单独成若干客户端（由 --neg_clients 控制）
    distribute   负样本均匀散到所有变化类客户端（会破坏单类原则）
    exclude      丢弃负样本

用法:
    python partition_by_class.py                                 # 默认 10 客户端/类 + class_like
    python partition_by_class.py --clients_per_class 5
    python partition_by_class.py --neg_strategy exclude
    python partition_by_class.py --dry_run
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

from partition_utils import (
    CLASS_NAMES,
    SOURCES,
    add_common_args,
    build_partition_output,
    get_sample_classes,
    parse_change_class,
    print_partition_summary,
    print_partition_summary_grouped,
    save_partition,
    scan_all_sources,
    split_list,
)


def partition_by_class(
    data_root: str,
    clients_per_class: int = 10,
    neg_strategy: str = "class_like",
    neg_clients: int | None = None,
    seed: int = 42,
) -> dict:
    """按语义类别划分（FedSeg 风格：每类拆分多客户端）。

    客户端 ID 遵循 FedSeg datasplit.py 公式：
        client_index = i + class_idx × clients_per_class
    其中 class_idx 是该类在 available_classes 排序后的索引，i 是该类内的子客户端序号。
    每个客户端只持有单一类别的样本（极端 Non-IID）。

    Args:
        data_root: 数据集根目录
        clients_per_class: 每个变化类别拆分成的客户端数
            （FedSeg Cityscapes 用 8，ADE20K 用 3；WHU-GCD 默认 10）
        neg_strategy: 负样本分配策略
            "class_like"  - 视为"类 0"，按 clients_per_class 切分（默认，最贴 FedSeg）
            "separate"    - 单独成若干客户端（由 neg_clients 控制数量）
            "distribute"  - 均匀散到所有变化类客户端（会破坏单类原则）
            "exclude"     - 丢弃负样本
        neg_clients: 当 neg_strategy="separate" 时的负样本客户端数。
            None 表示跟随 clients_per_class（仅对 class_like 生效）。
        seed: 随机种子
    """
    rng = random.Random(seed)

    print("正在扫描数据集...")
    source_data = scan_all_sources(data_root)
    print()

    # ─── 1. 收集 gcd 的带类别样本 ───
    gcd_samples = source_data.get("gcd", [])
    class_groups = get_sample_classes(gcd_samples)

    print("gcd 类别分布:")
    for cls_id in sorted(class_groups.keys()):
        name = CLASS_NAMES.get(cls_id, str(cls_id))
        print(f"  类别 {cls_id} ({name}): {len(class_groups[cls_id])} 样本")
    print()

    # ─── 2. 收集负样本 ───
    neg_samples = []
    for src in ["ucd", "ugcd", "ugcd_full"]:
        neg_samples.extend(source_data.get(src, []))
    print(f"负样本总数 (ucd+ugcd+ugcd_full): {len(neg_samples)}")
    print()

    # ─── 3. 容量检查：每类样本数是否足够均分给 clients_per_class 个客户端 ───
    available_classes = sorted(class_groups.keys())
    for cls_id in available_classes:
        n = len(class_groups[cls_id])
        if clients_per_class > n:
            print(f"  ⚠️ 警告: 类别 {cls_id} 只有 {n} 样本，不足以均分给 "
                  f"{clients_per_class} 个客户端（部分客户端将为空）")

    # ─── 4. FedSeg 风格划分：每类切 clients_per_class 份 ───
    # 客户端 ID 公式: client_index = i + class_idx × clients_per_class
    client_data = {}
    client_class_map = {}   # cid -> [cls_id]  （单元素列表，保持与旧版结构兼容）
    client_idx = 0

    for class_idx, cls_id in enumerate(available_classes):
        cls_samples = class_groups[cls_id]
        # split_list 内部会 shuffle 并尽量均匀切分（divmod）
        splits = split_list(cls_samples, clients_per_class, seed=seed + class_idx)
        for i, split in enumerate(splits):
            # 等价 FedSeg 公式: i + class_idx * clients_per_class
            # 但 client_idx 顺序累加更直观，且对后续 neg 追加友好
            assert client_idx == i + class_idx * clients_per_class, "ID 公式不一致"
            cid = f"client_{client_idx}"
            client_data[cid] = split
            client_class_map[cid] = [cls_id]
            client_idx += 1

    num_class_clients = client_idx
    print(f"变化类客户端: {num_class_clients} = {len(available_classes)} 类 × "
          f"{clients_per_class} 客户端/类")

    # ─── 5. 负样本处理 ───
    if not neg_samples:
        print("无负样本可分配")
    elif neg_strategy == "class_like":
        # 视为"类 0"，按 clients_per_class 切分（最贴 FedSeg 哲学）
        n_neg_clients = clients_per_class if neg_clients is None else neg_clients
        neg_splits = split_list(neg_samples, n_neg_clients, seed=seed + 1000)
        for i, split in enumerate(neg_splits):
            cid = f"client_{client_idx}"
            client_data[cid] = split
            client_class_map[cid] = [0]   # 用 0 标记负样本"类"
            client_idx += 1
        print(f"负样本客户端: {n_neg_clients} (class_like，每客户端 ~{len(neg_samples) // n_neg_clients} 样本)")
    elif neg_strategy == "separate":
        n_neg_clients = clients_per_class if neg_clients is None else neg_clients
        neg_splits = split_list(neg_samples, n_neg_clients, seed=seed + 1000)
        for i, split in enumerate(neg_splits):
            cid = f"client_{client_idx}"
            client_data[cid] = split
            client_class_map[cid] = [0]
            client_idx += 1
        print(f"负样本客户端: {n_neg_clients} (separate)")
    elif neg_strategy == "distribute":
        # 均匀散到所有变化类客户端（会破坏单类原则，仅用于对比实验）
        rng.shuffle(neg_samples)
        for i, s in enumerate(neg_samples):
            target_cid = f"client_{i % num_class_clients}"
            client_data[target_cid].append(s)
        print(f"负样本已均匀散到 {num_class_clients} 个变化类客户端 (distribute)")
    elif neg_strategy == "exclude":
        print("负样本已排除 (exclude)")
    else:
        raise ValueError(f"未知 neg_strategy: {neg_strategy}")

    class_names_str = ", ".join(
        f"{cid}: " + "+".join(CLASS_NAMES.get(c, str(c)) for c in cls_list)
        for cid, cls_list in list(client_class_map.items())[:6]
    )

    return build_partition_output(
        method="class",
        params={
            "clients_per_class": clients_per_class,
            "neg_strategy": neg_strategy,
            "neg_clients": (
                (clients_per_class if neg_clients is None else neg_clients)
                if neg_strategy in ("class_like", "separate") else 0
            ),
            "client_class_map": client_class_map,
            "style": "fedseg",
        },
        client_data=client_data,
    )


def main():
    parser = argparse.ArgumentParser(description="方案二：按语义类别划分（FedSeg 风格，每类拆分多客户端）")
    add_common_args(parser)
    parser.add_argument(
        "--clients_per_class",
        type=int,
        default=10,
        help="每个变化类别拆分成的客户端数 (FedSeg Cityscapes 用 8, WHU-GCD 默认 10)。"
             "总客户端数 ≈ num_classes × clients_per_class + 负样本客户端",
    )
    parser.add_argument(
        "--neg_strategy",
        type=str,
        default="class_like",
        choices=["class_like", "separate", "distribute", "exclude"],
        help="负样本策略: "
             "class_like=视为类0按 clients_per_class 切分 (推荐), "
             "separate=单独若干客户端, "
             "distribute=均匀散到变化类客户端, "
             "exclude=丢弃",
    )
    parser.add_argument(
        "--neg_clients",
        type=int,
        default=None,
        help="负样本客户端数 (仅 class_like/separate 生效，None=跟随 clients_per_class)",
    )
    args = parser.parse_args()

    result = partition_by_class(
        args.data_root,
        clients_per_class=args.clients_per_class,
        neg_strategy=args.neg_strategy,
        neg_clients=args.neg_clients,
        seed=args.seed,
    )

    # 客户端数较多时使用分组摘要（按主类别聚合），避免日志爆炸
    if result["num_clients"] > 20:
        print_partition_summary_grouped(result)
    else:
        print_partition_summary(result)

    if not args.dry_run:
        output_path = (
            f"{args.output_dir}/partition_class_fedseg"
            f"_cpc{args.clients_per_class}_{args.neg_strategy}.json"
        )
        save_partition(result, output_path)


if __name__ == "__main__":
    main()
