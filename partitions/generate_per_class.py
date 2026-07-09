"""
WHU-GCD 联邦学习「逐语义类」Non-IID 数据划分（极端 label-skew）。

把每个语义类作为独立单元，各自均匀拆成若干客户端 —— 每个客户端只
持有单一语义类的样本，构成最严格的类别异构（class-wise Non-IID）。

语义类（共 7 个）：
    0  无变化  ← ucd / ugcd / ugcd_full（负样本，作为独立语义类）
    2  建筑    ← gcd
    3  道路    ← gcd
    4  水体    ← gcd
    5  荒地    ← gcd
    6  森林    ← gcd
    7  农业    ← gcd

默认 7 类 × 10 客户端/类 = 70 个客户端；每类拆多少由 --clients_per_class
控制。本模块复用 partitions.generate 的扫描 / 统计 / 保存工具，产出
**相对路径** JSON（跨机器可移植，与现有 dirichlet 划分同 schema）。

用法（从仓库根目录 FedChange/ 运行）:
    # 默认：7 类 × 10 = 70 客户端
    python -m partitions.generate_per_class --data_root ../WHU-GCD

    # 自定义每类客户端数（如每类 5 个 → 35 客户端）
    python -m partitions.generate_per_class --clients_per_class 5 --data_root ../WHU-GCD

    # 仅预览统计，不写文件
    python -m partitions.generate_per_class --dry_run
"""

import argparse
import random

from partitions.generate import (
    CLASS_NAMES,
    add_common_args,
    build_partition_output,
    get_sample_classes,
    print_partition_summary,
    save_partition,
    scan_all_sources,
)


def _split_even(samples: list, n: int, seed: int = 42) -> list[list]:
    """将样本列表尽量均匀地拆成 n 份（先 shuffle 再等分）。

    与 partition_utils.split_list 行为一致，本模块自包含故内联实现。
    """
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    k, m = divmod(len(shuffled), n)
    return [
        shuffled[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        for i in range(n)
    ]


def partition_per_class(
    data_root: str,
    clients_per_class: int = 10,
    seed: int = 42,
) -> dict:
    """逐语义类划分（每个客户端只持有单一语义类）。

    Args:
        data_root: 数据集根目录
        clients_per_class: 每个语义类拆成的客户端数
            （7 类 × N = 7N 个客户端；默认 N=10 → 70 客户端）
        seed: 随机种子
    """
    print("正在扫描数据集...")
    source_data = scan_all_sources(data_root)
    print()

    # ─── 1. 变化类（gcd，类别 2-7）───
    gcd_samples = source_data.get("gcd", [])
    class_groups = get_sample_classes(gcd_samples)  # {2: [...], 3: [...], ...}

    # ─── 2. 未变化类（类 0）：负样本作为独立语义类 ───
    neg_samples: list[dict] = []
    for src in ["ucd", "ugcd", "ugcd_full"]:
        neg_samples.extend(source_data.get(src, []))

    # 组装 {class_id: samples}，类 0 在前
    all_class_groups: dict[int, list[dict]] = {0: neg_samples}
    all_class_groups.update(class_groups)

    # 打印各类分布
    print("语义类分布（每类将拆成 {} 个客户端）:".format(clients_per_class))
    for cls_id in sorted(all_class_groups.keys()):
        name = CLASS_NAMES.get(cls_id, str(cls_id))
        n = len(all_class_groups[cls_id])
        per_client = n // clients_per_class
        print(f"  类别 {cls_id} ({name}): {n} 样本 → ~{per_client}/客户端")
    print()

    # ─── 3. 每类均匀拆 clients_per_class 份，每份一个客户端 ───
    client_data: dict[str, list[dict]] = {}
    client_class_map: dict[str, int] = {}
    cid_idx = 0
    for cls_id in sorted(all_class_groups.keys()):
        samples = all_class_groups[cls_id]
        # 各类用独立种子，保证拆分互不影响且可复现
        splits = _split_even(samples, clients_per_class, seed=seed + cls_id)
        for split in splits:
            cid = f"client_{cid_idx}"
            client_data[cid] = split
            client_class_map[cid] = cls_id
            cid_idx += 1

    return build_partition_output(
        method="per_class",
        params={
            "clients_per_class": clients_per_class,
            "num_classes": len(all_class_groups),
            "client_class_map": client_class_map,
        },
        client_data=client_data,
    )


def main():
    parser = argparse.ArgumentParser(
        description="逐语义类划分（每客户端单一语义类，极端 label-skew Non-IID）"
    )
    add_common_args(parser)
    parser.add_argument(
        "--clients_per_class",
        type=int,
        default=10,
        help="每个语义类拆成的客户端数（7 类 × N；默认 10 → 70 客户端）",
    )
    args = parser.parse_args()

    result = partition_per_class(
        args.data_root,
        clients_per_class=args.clients_per_class,
        seed=args.seed,
    )

    print_partition_summary(result)

    if not args.dry_run:
        total = result["num_clients"]
        output_path = (
            f"{args.output_dir}/partition_perclass"
            f"_c{args.clients_per_class}_n{total}.json"
        )
        save_partition(result, output_path)


if __name__ == "__main__":
    main()
