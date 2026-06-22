#!/bin/bash
# Federated torchange baselines on WHU-GCD (binary change detection).
# Federates each torchange baseline with FedAvg and FedProx under the SAME Non-IID
# partition and schedule as the BIT-CD federated experiments (see run_fed_bcd.sh).
#
# The FedChange aggregation is model-agnostic (state_dict averaging), so torchange
# models (wrapped by TorchangeCDAdapter) plug into fed_main unchanged. Each client
# trains with the model's NATIVE loss (BCE+Dice).
#
# Run from FedChange/ directory:
#   bash scripts/run_fed_torchange_bcd.sh
#
# changen2_zeroshot is eval-only and run separately.

EPOCHS=${1:-200}
FRAC_NUM=${2:-5}
LOCAL_EP=${3:-2}
BATCH_SIZE=${4:-8}
IMG_SIZE=${5:-256}
LR=${6:-0.01}
FEDPROX_MU=${7:-0.01}

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
    "dirichlet_05 partitions/partition_dirichlet_a0.5_n7.json"
    # "dirichlet_01 partitions/partition_dirichlet_a0.1_n7.json"
    # "source       partitions/partition_source.json"
    # "iid          partitions/partition_dirichlet_a100.0_n7.json"
)

for NET_G in "${MODELS[@]}"; do
    for entry in "${PARTITIONS[@]}"; do
        name=$(echo "$entry" | cut -d' ' -f1)
        file=$(echo "$entry" | cut -d' ' -f2)

        # FedAvg (fedprox_mu = 0)
        proj_avg="FedAvg_${NET_G}_${name}_bcd"
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
            --fedprox_mu 0.0 \
            --eval_splits "val,test,test2" \
            --global_test_frequency 20 \
            --save_frequency 20 \
            --project_name "$proj_avg" \
            --checkpoint_root "results/torchange_fed" \
            --seed 42

        # FedProx
        proj_prox="FedProx_${NET_G}_${name}_bcd"
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
            --fedprox_mu "$FEDPROX_MU" \
            --eval_splits "val,test,test2" \
            --global_test_frequency 20 \
            --save_frequency 20 \
            --project_name "$proj_prox" \
            --checkpoint_root "results/torchange_fed" \
            --seed 42
    done
done

echo ""
echo "========== All federated torchange baselines complete =========="
