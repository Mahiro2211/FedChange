"""
按语义类别的 Non-IID 划分（FedSeg 类别异构扩展版）。

参考 FedSeg (CVPR 2023) 的类别异构思想：每个客户端只持有部分语义类别的数据，
通过 classes_per_client 控制异构程度：

    Non-IID1 (classes_per_client=1): 每个客户端只含 1 个语义类
    Non-IID2 (classes_per_client=2): 每个客户端含 2 个语义类

**类 0（无变化）作为独立的第 7 个语义类**，与 6 个变化类 {2..7} 一视同仁地参与划分。
因此 WHU-GCD 共 7 个语义类：{0 无变化, 2 建筑, 3 道路, 4 水体, 5 荒地, 6 森林, 7 农业}。
负样本 (ucd/ugcd/ugcd_full) 即类 0，不再单独"均撒"，而是像其他类一样分配给其归属客户端。

与原 FedSeg（K = 类别数 × 每类客户端数，K 随设置变化）不同，本脚本固定总客户端数 K，
使 Non-IID1 / Non-IID2 / IID 三者客户端数与客户端尺寸一致，唯一变量是异构度。

类别分配采用"循环窗口"：7 个语义类 {0,2,3,4,5,6,7} 排成环，客户端 j 持有连续
classes_per_client 个类
    assigned_classes[j] = [ C[(j+t) % num_classes] for t in range(classes_per_client) ]
该分配对任意 classes_per_client、K 都均衡：每个类恰好出现在
    classes_per_client × K / num_classes 个客户端中。

注意：Non-IID1 会出现"只含类 0（纯无变化）"的客户端——这是忠实反映极端类异构的场景
（某些区域观测期内确实无任何变化），是研究的意义所在，非缺陷。

用法:
    # Non-IID1: 70 客户端, 每客户端 1 类（7 类 × 10 客户端/类）
    python partition_noniid.py --classes_per_client 1 --num_clients 70

    # Non-IID2: 70 客户端, 每客户端 2 类
    python partition_noniid.py --classes_per_client 2 --num_clients 70

    # 预览不写文件
    python partition_noniid.py --classes_per_client 1 --dry_run
"""

import argparse

from partition_utils import (
    CLASS_NAMES,
    add_common_args,
    build_partition_output,
    get_sample_classes,
    print_partition_summary_grouped,
    save_partition,
    scan_all_sources,
    split_list,
)


def partition_class_heterogeneous(
    data_root: str,
    num_clients: int = 70,
    classes_per_client: int = 1,
    seed: int = 42,
) -> dict:
    """按语义类别的 Non-IID 划分（固定总客户端数 K，循环窗口类别分配，含类 0）。

    类 0（无变化 / 负样本）作为独立语义类与变化类 {2..7} 一同参与划分。

    Args:
        data_root: 数据集根目录
        num_clients: 总客户端数 K（Non-IID1/Non-IID2/IID 应使用相同 K 以公平对比；
            建议为类别数 7 的倍数，默认 70 = 7 类 × 10 客户端/类）
        classes_per_client: 每个客户端持有的语义类数（1=Non-IID1, 2=Non-IID2）
            必须 ≤ 语义类总数 (7)
        seed: 随机种子
    """
    print("正在扫描数据集...")
    source_data = scan_all_sources(data_root)
    print()

    # ─── 1. 收集 6 个变化类（gcd） ───
    gcd_samples = source_data.get("gcd", [])
    class_groups = get_sample_classes(gcd_samples)  # {2:[...], ..., 7:[...]}

    # ─── 2. 把负样本作为类 0 并入类别池（第 7 个语义类）───
    neg_samples = []
    for src in ["ucd", "ugcd", "ugcd_full"]:
        neg_samples.extend(source_data.get(src, []))
    class_groups[0] = neg_samples

    available_classes = sorted(class_groups.keys())  # [0, 2, 3, 4, 5, 6, 7]
    num_classes = len(available_classes)

    print(f"语义类分布（共 {num_classes} 类，含类 0 无变化）:")
    for cls_id in available_classes:
        name = CLASS_NAMES.get(cls_id, str(cls_id))
        print(f"  类别 {cls_id} ({name}): {len(class_groups[cls_id])} 样本")
    print()

    if classes_per_client > num_classes:
        raise ValueError(
            f"classes_per_client={classes_per_client} 超过语义类总数 {num_classes}"
        )
    if num_clients % num_classes != 0:
        print(f"  ⚠️ 警告: num_clients={num_clients} 不是类别数 {num_classes} 的倍数，"
              f"各类的客户端数可能不均（建议用 {num_classes} 的倍数）")

    # ─── 3. 循环窗口类别分配 ───
    # client j -> classes [C[(j+t) % num_classes] for t in range(classes_per_client)]
    assigned_classes: list[list[int]] = []
    for j in range(num_clients):
        cls = [available_classes[(j + t) % num_classes] for t in range(classes_per_client)]
        assigned_classes.append(cls)

    holders_per_class = {c: [] for c in available_classes}
    for j, cls_list in enumerate(assigned_classes):
        for c in cls_list:
            holders_per_class[c].append(j)

    print(f"划分: K={num_clients}, classes_per_client={classes_per_client}")
    for c in available_classes:
        n_holders = len(holders_per_class[c])
        per_holder = len(class_groups[c]) // max(n_holders, 1)
        print(f"  类别 {c} ({CLASS_NAMES.get(c, c)}): {n_holders} 个客户端, "
              f"每客户端 ~{per_holder} 样本")
    print()

    # ─── 4. 每类样本在其所属客户端间均分（类 0 与变化类一视同仁）───
    client_data: dict[str, list] = {f"client_{j}": [] for j in range(num_clients)}
    for c in available_classes:
        holders = holders_per_class[c]
        chunks = split_list(class_groups[c], len(holders), seed=seed + c)
        for j, chunk in zip(holders, chunks):
            client_data[f"client_{j}"].extend(chunk)

    return build_partition_output(
        method="class",
        params={
            "style": "noniid",
            "classes_per_client": classes_per_client,
            "num_clients": num_clients,
            "num_semantic_classes": num_classes,
            "class0_as_class": True,
            "assigned_classes": {f"client_{j}": assigned_classes[j] for j in range(num_clients)},
        },
        client_data=client_data,
    )


def main():
    parser = argparse.ArgumentParser(
        description="按语义类别的 Non-IID 划分（含类 0，classes_per_client 1/2, 固定 K）"
    )
    add_common_args(parser)
    parser.add_argument(
        "--classes_per_client",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="每个客户端持有的语义类数（1=Non-IID1, 2=Non-IID2）",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=70,
        help="总客户端数 K（建议 Non-IID1/2/IID 用相同 K，默认 70 = 7 类 × 10 客户端/类）",
    )
    args = parser.parse_args()

    result = partition_class_heterogeneous(
        args.data_root,
        num_clients=args.num_clients,
        classes_per_client=args.classes_per_client,
        seed=args.seed,
    )

    # 客户端数较多时按主类别分组打印，避免日志爆炸
    if result["num_clients"] > 20:
        print_partition_summary_grouped(result)
    else:
        from partition_utils import print_partition_summary
        print_partition_summary(result)

    if not args.dry_run:
        tag = "noniid" + str(args.classes_per_client)
        output_path = f"{args.output_dir}/partition_{tag}_K{args.num_clients}.json"
        save_partition(result, output_path)


if __name__ == "__main__":
    main()
