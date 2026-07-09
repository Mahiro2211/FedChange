#!/bin/bash
# Federated torchange baselines on WHU-GCD (binary change detection).
# Federates each torchange baseline with FedAvg and FedProx under the SAME Non-IID
# partition and schedule as the BIT-CD federated experiments (see run_fed_bcd.sh),
# repeated over multiple seeds for mean±std reporting.
#
# The FedChange aggregation is model-agnostic (state_dict averaging), so torchange
# models (wrapped by TorchangeCDAdapter) plug into fed_main unchanged. Each client
# trains with the model's NATIVE loss (BCE+Dice).
#
# Run from FedChange/ directory:
#   bash scripts/run_fed_torchange_bcd.sh
# Custom seeds:
#   bash scripts/run_fed_torchange_bcd.sh 200 5 2 8 256 0.01 0.01 "42 2024 0"
#
# changen2_zeroshot is eval-only and run separately.

EPOCHS=${1:-200}
FRAC_NUM=${2:-5}
LOCAL_EP=${3:-2}
BATCH_SIZE=${4:-8}
IMG_SIZE=${5:-256}
LR=${6:-0.01}
FEDPROX_MU=${7:-0.01}
SEEDS_STR=${8:-"42 2024 0"}

# ─── 环境依赖检查（torchange 基线实验，核心库 + torchange 可选库）───
# shellcheck source=check_env.sh
source "$(dirname "$0")/check_env.sh"
check_env_core || exit 1
check_env_torchange || exit 1

read -ra SEEDS <<< "$SEEDS_STR"

# Trainable torchange BCD baselines
MODELS=(
    "changesparse_bcd"
    "changestar_1xd_r18"
    "changestar_1xd"
    "changestar_2_5"
)

# Use the canonical Non-IID partition (Dirichlet alpha=0.5, 7 clients) for the
# main comparison table. Add more partition entries below to sweep heterogeneity.
PARTITIONS=(
    "dirichlet_05  partitions/partition_dirichlet_a0.5_n7.json"
    # "dirichlet_01  partitions/partition_dirichlet_a0.1_n7.json"
    # "dirichlet_100 partitions/partition_dirichlet_a100.0_n7.json"
)

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "========== Seed = $seed =========="
    for NET_G in "${MODELS[@]}"; do
        for entry in "${PARTITIONS[@]}"; do
            read -r name file <<< "$entry"

            # FedAvg (fed_alg=fedavg)
            proj_avg="FedAvg_${NET_G}_${name}_bcd_s${seed}"
            echo ""
            echo ">>> $proj_avg"
            python -m fed_cd.federated.fed_main \
                --partition_json "$file" \
                --data_root ../WHU-GCD \
                --net_G "$NET_G" \
                --num_classes 2 \
                --img_size "$IMG_SIZE" \
                --epochs "$EPOCHS" \
                --frac_num "$FRAC_NUM" \
                --local_ep "$LOCAL_EP" \
                --local_bs "$BATCH_SIZE" \
                --lr "$LR" \
                --lr_policy linear \
                --optimizer sgd \
                --pretrained True \
                --iid False \
                --fed_alg fedavg --fedprox_mu 0.0 \
                --eval_splits "val,test,test2" \
                --global_test_frequency 20 \
                --save_frequency 20 \
                --project_name "$proj_avg" \
                --checkpoint_root "results/torchange_fed" \
                --seed "$seed"

            # FedProx (fed_alg=fedprox)
            proj_prox="FedProx_${NET_G}_${name}_bcd_s${seed}"
            echo ""
            echo ">>> $proj_prox"
            python -m fed_cd.federated.fed_main \
                --partition_json "$file" \
                --data_root ../WHU-GCD \
                --net_G "$NET_G" \
                --num_classes 2 \
                --img_size "$IMG_SIZE" \
                --epochs "$EPOCHS" \
                --frac_num "$FRAC_NUM" \
                --local_ep "$LOCAL_EP" \
                --local_bs "$BATCH_SIZE" \
                --lr "$LR" \
                --lr_policy linear \
                --optimizer sgd \
                --pretrained True \
                --iid False \
                --fed_alg fedprox --fedprox_mu "$FEDPROX_MU" \
                --eval_splits "val,test,test2" \
                --global_test_frequency 20 \
                --save_frequency 20 \
                --project_name "$proj_prox" \
                --checkpoint_root "results/torchange_fed" \
                --seed "$seed"
        done
    done
done

echo ""
echo "========== All federated torchange baselines complete =========="
echo "汇总: python -m fed_cd.summarize --results_root results/torchange_fed"
