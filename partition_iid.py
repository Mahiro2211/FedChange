"""
分层随机 IID 划分（Non-IID1/Non-IID2 的对比基线）。

为每个客户端按整体类别比例分层随机采样：每个客户端都包含全部 6 个变化类 + 负样本，
且类别分布与全局一致 → 真正的 IID。

与 Non-IID1/Non-IID2 使用相同的总客户端数 K 和相近的客户端尺寸，使三者唯一变量是
数据异构度，从而公平对比。

用法:
    # 70 客户端分层 IID（与 Non-IID1/2 的 K=70 对应）
    python partition_iid.py --num_clients 70

    # 预览不写文件
    python partition_iid.py --num_clients 70 --dry_run
"""

import argparse

from partition_utils import (
    CLASS_NAMES,
    add_common_args,
    build_partition_output,
    get_sample_classes,
    print_partition_summary,
    save_partition,
    scan_all_sources,
    split_list,
)


def partition_iid(data_root: str, num_clients: int = 70, seed: int = 42) -> dict:
    """分层随机 IID 划分。

    对每个类别（含负样本作为一个组）独立 shuffle 后均分成 num_clients 份，
    每个客户端从每个类拿到一份 → 每客户端都含全部类别，分布与全局一致。

    Args:
        data_root: 数据集根目录
        num_clients: 客户端数 K（应与 Non-IID1/2 一致）
        seed: 随机种子
    """
    print("正在扫描数据集...")
    source_data = scan_all_sources(data_root)
    print()

    gcd_samples = source_data.get("gcd", [])
    class_groups = get_sample_classes(gcd_samples)

    print("gcd 类别分布:")
    for cls_id in sorted(class_groups.keys()):
        name = CLASS_NAMES.get(cls_id, str(cls_id))
        print(f"  类别 {cls_id} ({name}): {len(class_groups[cls_id])} 样本")
    print()

    neg_samples = []
    for src in ["ucd", "ugcd", "ugcd_full"]:
        neg_samples.extend(source_data.get(src, []))
    print(f"负样本总数 (ucd+ugcd+ugcd_full): {len(neg_samples)}")
    print()

    # 每个 client 累积来自各类的分层切片
    client_data: dict[str, list] = {f"client_{j}": [] for j in range(num_clients)}

    def _distribute(group_samples, group_seed):
        if not group_samples:
            return
        chunks = split_list(group_samples, num_clients, seed=group_seed)
        for j, chunk in enumerate(chunks):
            client_data[f"client_{j}"].extend(chunk)

    # 每个变化类分层
    for cls_id in sorted(class_groups.keys()):
        _distribute(class_groups[cls_id], seed + cls_id)
    # 负样本分层
    _distribute(neg_samples, seed + 1000)

    print(f"IID 划分: K={num_clients} 客户端, 分层随机（每客户端含全部类别）")

    return build_partition_output(
        method="iid",
        params={
            "style": "stratified",
            "num_clients": num_clients,
        },
        client_data=client_data,
    )


def main():
    parser = argparse.ArgumentParser(description="分层随机 IID 划分")
    add_common_args(parser)
    parser.add_argument(
        "--num_clients",
        type=int,
        default=70,
        help="客户端数 K（应与 Non-IID1/2 一致，默认 70 = 7 类 × 10）",
    )
    args = parser.parse_args()

    result = partition_iid(args.data_root, num_clients=args.num_clients, seed=args.seed)

    print_partition_summary(result)

    if not args.dry_run:
        output_path = f"{args.output_dir}/partition_iid_K{args.num_clients}.json"
        save_partition(result, output_path)


if __name__ == "__main__":
    main()
