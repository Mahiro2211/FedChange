#!/bin/bash
# Federated BCD experiments: FedAvg and FedProx on a sweep of Dirichlet Non-IID
# partitions (controlled heterogeneity via alpha), repeated over multiple random
# seeds for mean±std reporting.
#
# project_name embeds the seed (e.g. FedAvg_dirichlet_05_bcd_s42) so each seed's
# results.json lands in its own dir and is picked up by summarize's seed table.
#
# Run from FedChange/ directory:
#   bash scripts/run_fed_bcd.sh
# Custom seeds / hyperparams (positional):
#   bash scripts/run_fed_bcd.sh 200 5 2 8 256 0.01 0.01 "42 2024 0"

EPOCHS=${1:-200}
FRAC_NUM=${2:-5}
LOCAL_EP=${3:-2}
BATCH_SIZE=${4:-8}
IMG_SIZE=${5:-256}
LR=${6:-0.01}
FEDPROX_MU=${7:-0.01}
SEEDS_STR=${8:-"42 2024 0"}

# ─── 环境依赖检查（BIT-CD 实验，仅需核心库）───
# shellcheck source=check_env.sh
source "$(dirname "$0")/check_env.sh"
check_env_core || exit 1

read -ra SEEDS <<< "$SEEDS_STR"

# Partition files: Dirichlet alpha sweep (smaller alpha = more heterogeneous).
# alpha=0.1 极端异构 / 0.5 中等 / 1.0 轻度 / 100 接近 IID
# 生成划分: python -m partitions.generate --alpha <α> --num_clients 7 --data_root ../WHU-GCD
PARTITIONS=(
    "dirichlet_01   partitions/partition_dirichlet_a0.1_n7.json"
    "dirichlet_05   partitions/partition_dirichlet_a0.5_n7.json"
    "dirichlet_10   partitions/partition_dirichlet_a1.0_n7.json"
    "dirichlet_100  partitions/partition_dirichlet_a100.0_n7.json"
)

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
        "--checkpoint_root" "results/fed_bcd"
        "--seed" "$seed"
    )

    echo ""
    echo "========== Seed = $seed =========="

    # FedAvg experiments (fed_alg=fedavg, fedprox_mu=0)
    echo "--- FedAvg (seed=$seed) ---"
    for entry in "${PARTITIONS[@]}"; do
        read -r name file <<< "$entry"
        proj_name="FedAvg_${name}_bcd_s${seed}"
        echo ">>> $proj_name"
        python -m fed_cd.federated.fed_main \
            --partition_json "$file" \
            --project_name "$proj_name" \
            --fed_alg fedavg --fedprox_mu 0.0 --iid False \
            "${COMMON[@]}"
    done

    # FedProx experiments (fed_alg=fedprox)
    echo "--- FedProx mu=$FEDPROX_MU (seed=$seed) ---"
    for entry in "${PARTITIONS[@]}"; do
        read -r name file <<< "$entry"
        proj_name="FedProx${FEDPROX_MU}_${name}_bcd_s${seed}"
        echo ">>> $proj_name"
        python -m fed_cd.federated.fed_main \
            --partition_json "$file" \
            --project_name "$proj_name" \
            --fed_alg fedprox --fedprox_mu "$FEDPROX_MU" --iid False \
            "${COMMON[@]}"
    done
done

echo ""
echo "========== All BCD experiments complete =========="
echo "汇总: python -m fed_cd.summarize --results_root results/fed_bcd"
