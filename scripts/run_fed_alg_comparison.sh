#!/bin/bash
# Federated algorithm comparison: FedAvg vs FedProx vs FedNova vs SCAFFOLD.
#
# This is the core P0 SOTA-comparison experiment (addresses "缺少 SOTA 联邦学习算法对比").
# On a fixed Non-IID partition, sweep all four algorithms × multiple seeds so the
# main table reports mean±std across algorithms. Default partition is Dirichlet
# alpha=0.5 (moderate heterogeneity, where algorithm differences are most visible).
#
# Note on SCAFFOLD + torchange: SCAFFOLD degrades to FedAvg for torchange models
# (black-box loss); a warning is logged. This script defaults to BIT-CD where all
# four algorithms are fully supported.
#
# Run from FedChange/ directory:
#   bash scripts/run_fed_alg_comparison.sh
# Custom:
#   bash scripts/run_fed_alg_comparison.sh partitions/partition_dirichlet_a0.5_n7.json "42 2024 0"

PARTITION=${1:-partitions/partition_dirichlet_a0.5_n7.json}
SEEDS_STR=${2:-"42 2024 0"}
EPOCHS=${3:-200}
FRAC_NUM=${4:-5}
LOCAL_EP=${5:-2}
BATCH_SIZE=${6:-8}
IMG_SIZE=${7:-256}
LR=${8:-0.01}
FEDPROX_MU=${9:-0.01}

TAG=$(basename "$PARTITION" | sed -E 's/^partition_(.+)\.json$/\1/')
read -ra SEEDS <<< "$SEEDS_STR"

if [ ! -f "$PARTITION" ]; then
    echo "⚠️  缺少 $PARTITION，请先生成划分文件。"
    exit 1
fi

# 4 algorithms
ALGS=("fedavg" "fedprox" "fednova" "scaffold")

for seed in "${SEEDS[@]}"; do
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
        "--checkpoint_root" "results/alg_comparison"
        "--partition_json" "$PARTITION"
        "--iid" "False"
        "--seed" "$seed"
    )
    echo ""
    echo "========== Seed = $seed (partition=$TAG) =========="

    for alg in "${ALGS[@]}"; do
        proj="${alg}_${TAG}_bcd_s${seed}"
        echo ""
        echo ">>> $proj"
        # FedProx needs mu; others pass mu=0 (ignored unless fed_alg=fedprox).
        python -m fed_cd.federated.fed_main \
            --fed_alg "$alg" \
            --fedprox_mu "$FEDPROX_MU" \
            --project_name "$proj" \
            "${COMMON[@]}"
    done
done

echo ""
echo "========== Algorithm comparison complete =========="
echo "汇总: python -m fed_cd.summarize --results_root results/alg_comparison"
