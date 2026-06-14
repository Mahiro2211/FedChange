"""
方案一：按数据来源划分（模拟域偏移）

将 WHU-GCD 的 4 种训练数据来源（gcd / ugcd_full / ucd / ugcd）
分配给不同客户端，利用合成 vs 真实数据的天然域差异模拟 Non-IID。

用法:
    python partition_by_source.py
    python partition_by_source.py --clients_per_source 2,1,1,1
    python partition_by_source.py --dry_run
"""

import argparse
import sys

from partition_utils import (
    SOURCES,
    add_common_args,
    build_partition_output,
    print_partition_summary,
    save_partition,
    scan_all_sources,
    split_list,
)


def partition_by_source(
    data_root: str,
    clients_per_source: list[int],
    seed: int = 42,
) -> dict:
    """按数据来源划分。

    Args:
        data_root: WHU-GCD 数据集根目录
        clients_per_source: 各来源的客户端数，顺序对应 [gcd, ugcd_full, ucd, ugcd]
        seed: 随机种子
    """
    assert len(clients_per_source) == len(SOURCES), (
        f"clients_per_source 长度 ({len(clients_per_source)}) "
        f"必须等于来源数 ({len(SOURCES)})"
    )

    print("正在扫描数据集...")
    source_data = scan_all_sources(data_root)
    print()

    client_data = {}
    client_idx = 0

    for src_name, num_clients in zip(SOURCES, clients_per_source):
        samples = source_data.get(src_name, [])
        if not samples:
            print(f"  跳过空来源: {src_name}")
            continue

        # 将该来源的数据均匀分成 num_clients 份
        splits = split_list(samples, num_clients, seed=seed + client_idx)

        for split in splits:
            cid = f"client_{client_idx}"
            client_data[cid] = split
            client_idx += 1

        print(f"  {src_name}: {len(samples)} 样本 → {num_clients} 客户端"
              f" (每客户端 ~{len(samples) // num_clients})")

    return build_partition_output(
        method="source",
        params={"clients_per_source": dict(zip(SOURCES, clients_per_source))},
        client_data=client_data,
    )


def main():
    parser = argparse.ArgumentParser(description="方案一：按数据来源划分")
    add_common_args(parser)
    parser.add_argument(
        "--clients_per_source",
        type=str,
        default="3,2,1,1",
        help="各来源客户端数，逗号分隔，顺序: gcd,ugcd_full,ucd,ugcd (默认: 3,2,1,1)",
    )
    args = parser.parse_args()

    cps = [int(x) for x in args.clients_per_source.split(",")]
    result = partition_by_source(args.data_root, cps, seed=args.seed)

    print_partition_summary(result)

    if not args.dry_run:
        output_path = f"{args.output_dir}/partition_source.json"
        save_partition(result, output_path)


if __name__ == "__main__":
    main()
