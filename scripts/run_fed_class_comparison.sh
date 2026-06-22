#!/bin/bash
# Non-IID 类别异构对比实验（FedSeg 风格扩展）：Non-IID1 / Non-IID2 / IID
#
# 三类划分用相同客户端数 K=70、相近客户端尺寸（均值≈390），唯一变量是数据异构度：
#   Non-IID1: 每客户端 1 个语义类（7 类含类0 × 10 客户端/类，含纯无变化客户端）
#   Non-IID2: 每客户端 2 个语义类（每类出现在 20 个客户端）
#   IID:      分层随机，每客户端含全部 7 类
#
# 每类划分 × (FedAvg + FedProx) = 6 个实验。无 centralized。
# 生成划分文件需先运行：
#   python partition_noniid.py --classes_per_client 1 --num_clients 70
#   python partition_noniid.py --classes_per_client 2 --num_clients 70
#   python partition_iid.py --num_clients 70
#
# Run from FedChange/ directory:
#   bash scripts/run_fed_class_comparison.sh

EPOCHS=${1:-200}
FRAC_NUM=${2:-5}
LOCAL_EP=${3:-2}
BATCH_SIZE=${4:-8}
IMG_SIZE=${5:-256}
LR=${6:-0.01}
FEDPROX_MU=${7:-0.01}
K=${8:-70}

COMMON=(
    "--data_root" "../WHU-GCD"
    "--net_G" "base_transformer_pos_s4_dd8"
    "--num_classes" "2"
    "--img_size" "$IMG_SIZE"
    "--epochs" "$EPOCHS"
    "--frac_num" "$FRAC_NUM"
    "--local_ep" "$LOCAL_EP"
    "--local_bs" "$BATCH_SIZE"
    "--lr" "$LR"
    "--lr_policy" "linear"
    "--optimizer" "sgd"
    "--pretrained" "True"
    "--eval_splits" "val,test,test2"
    "--global_test_frequency" "20"
    "--save_frequency" "20"
    "--checkpoint_root" "results/class_comparison"
    "--seed" "42"
)

# 划分: tag | json | --iid (IID=简单平均, Non-IID=加权平均, 遵循 FedSeg 惯例)
PARTITIONS=(
    "noniid1 partitions/partition_noniid1_K${K}.json False"
    "noniid2 partitions/partition_noniid2_K${K}.json False"
    "iid     partitions/partition_iid_K${K}.json True"
)

for entry in "${PARTITIONS[@]}"; do
    tag=$(echo "$entry" | cut -d' ' -f1)
    file=$(echo "$entry" | cut -d' ' -f2)
    iid=$(echo "$entry" | cut -d' ' -f3)

    if [ ! -f "$file" ]; then
        echo "⚠️  缺少 $file，请先生成划分（见脚本头注释）。跳过 $tag。"
        continue
    fi

    # FedAvg (fedprox_mu = 0)
    proj="FedAvg_${tag}_K${K}_bcd"
    echo ""
    echo ">>> $proj"
    python -m fed_cd.federated.fed_main \
        --partition_json "$file" --project_name "$proj" \
        --fedprox_mu 0.0 --iid "$iid" "${COMMON[@]}"

    # FedProx
    proj="FedProx${FEDPROX_MU}_${tag}_K${K}_bcd"
    echo ""
    echo ">>> $proj"
    python -m fed_cd.federated.fed_main \
        --partition_json "$file" --project_name "$proj" \
        --fedprox_mu "$FEDPROX_MU" --iid "$iid" "${COMMON[@]}"
done

echo ""
echo "========== Non-IID 类别对比实验完成 =========="
