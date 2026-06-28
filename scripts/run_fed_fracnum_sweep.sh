#!/bin/bash
# frac_num (客户端参与率) 稳健性 sweep 实验
#
# 在同一 partition 上 sweep 多个 frac_num 值，验证联邦变化检测结论对客户端参与率
# 是否稳健（弱点 4：单一低参与率 7% 可能放大 Non-IID 负面影响）。
# 默认在强异构 Non-IID1 (K=70) 上 sweep，frac_num 影响最显著。
#
# 默认 frac_num 序列 [1, 5, 10, 20, 70(全选)] 覆盖 1.4% ~ 100% 参与率。
# 注：fed_main.py 用 min(frac_num, num_clients) 处理，frac_num>=K 即全选。
#
# 仅跑 FedAvg (fedprox_mu=0)，因为目的是验证参与率稳健性而非算法对比。
# 每个 frac_num 一个独立 project_name，避免覆盖。
#
# Run from FedChange/ directory:
#   bash scripts/run_fed_fracnum_sweep.sh
# 自定义 partition 和 frac_num 序列：
#   bash scripts/run_fed_fracnum_sweep.sh partitions/partition_noniid1_K70.json "1 5 10 20 70"

# ─── 可调参数 ───
PARTITION=${1:-partitions/partition_noniid1_K70.json}
FRAC_LIST_STR=${2:-"1 5 10 20 70"}   # 空格分隔的 frac_num 值
EPOCHS=${3:-200}
LOCAL_EP=${4:-2}
BATCH_SIZE=${5:-8}
IMG_SIZE=${6:-256}
LR=${7:-0.01}

# 从 partition 文件名解析 tag（如 partition_noniid1_K70.json -> noniid1_K70）
TAG=$(basename "$PARTITION" | sed -E 's/^partition_(.+)\.json$/\1/')

# 检查 partition 文件存在
if [ ! -f "$PARTITION" ]; then
    echo "⚠️  缺少 $PARTITION，请先生成划分文件。"
    echo "    Non-IID1: python partition_noniid.py --classes_per_client 1 --num_clients 70"
    exit 1
fi

# 从 JSON 读取客户端数 K（用于校验 frac_num <= K）
K=$(python -c "import json; print(json.load(open('$PARTITION'))['num_clients'])" 2>/dev/null || echo "70")
echo "Partition: $PARTITION (K=$K, tag=$TAG)"
echo "frac_num sweep values: $FRAC_LIST_STR"
echo ""

# COMMON_ARGS（不含 frac_num，frac_num 每个 run 独立设置）
COMMON=(
    "--data_root" "../WHU-GCD"
    "--net_G" "base_transformer_pos_s4_dd8"
    "--num_classes" "2"
    "--img_size" "$IMG_SIZE"
    "--epochs" "$EPOCHS"
    "--local_ep" "$LOCAL_EP"
    "--local_bs" "$BATCH_SIZE"
    "--lr" "$LR"
    "--lr_policy" "linear"
    "--optimizer" "sgd"
    "--pretrained" "True"
    "--eval_splits" "val,test,test2"
    "--global_test_frequency" "20"
    "--save_frequency" "20"
    "--checkpoint_root" "results/fracnum_sweep"
    "--iid" "False"
    "--seed" "42"
)

echo "========== frac_num sweep on $TAG (K=$K) =========="
# 将空格分隔的 frac_num 列表转为数组
read -ra FRAC_LIST <<< "$FRAC_LIST_STR"
for frac in "${FRAC_LIST[@]}"; do
    # fed_main 内部会做 min(frac, K)，这里仅做提示
    if [ "$frac" -ge "$K" ]; then
        pct="100.0"
    else
        pct=$(python -c "print(f'{$frac/$K*100:.1f}')")
    fi
    proj="FedAvg_frac${frac}_${TAG}_bcd"
    echo ""
    echo ">>> $proj  (frac_num=$frac, 参与率 ${pct}%)"
    python -m fed_cd.federated.fed_main \
        --partition_json "$PARTITION" \
        --project_name "$proj" \
        --fedprox_mu 0.0 \
        --frac_num "$frac" \
        "${COMMON[@]}"
done

echo ""
echo "========== frac_num sweep 完成 =========="
echo "汇总: python -m fed_cd.summarize --results_root results/fracnum_sweep"
