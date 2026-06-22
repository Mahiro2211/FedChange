#!/bin/bash
# Federated BCD experiments: FedAvg and FedProx on multiple Non-IID partitions
# Run from FedChange/ directory:
#   bash scripts/run_fed_bcd.sh

EPOCHS=${1:-200}
FRAC_NUM=${2:-5}
LOCAL_EP=${3:-2}
BATCH_SIZE=${4:-8}
IMG_SIZE=${5:-256}
LR=${6:-0.01}
FEDPROX_MU=${7:-0.01}

COMMON_ARGS=(
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
    "--seed" "42"
)

# Partition files
PARTITIONS=(
    "source partitions/partition_source.json"
    "dirichlet_01 partitions/partition_dirichlet_a0.1_n7.json"
    "dirichlet_05 partitions/partition_dirichlet_a0.5_n7.json"
    "dirichlet_10 partitions/partition_dirichlet_a1.0_n7.json"
    "iid partitions/partition_dirichlet_a100.0_n7.json"
    "hybrid partitions/partition_hybrid_c1_separate.json"
)

# FedAvg experiments (fedprox_mu=0)
echo ""
echo "========== FedAvg Experiments =========="
for entry in "${PARTITIONS[@]}"; do
    name=$(echo "$entry" | cut -d' ' -f1)
    file=$(echo "$entry" | cut -d' ' -f2)
    proj_name="FedAvg_${name}_bcd"
    echo ""
    echo ">>> Running $proj_name ..."
    python -m fed_cd.federated.fed_main \
        --partition_json "$file" \
        --project_name "$proj_name" \
        --fedprox_mu 0.0 \
        "${COMMON_ARGS[@]}"
done

# FedProx experiments
echo ""
echo "========== FedProx Experiments (mu=$FEDPROX_MU) =========="
for entry in "${PARTITIONS[@]}"; do
    name=$(echo "$entry" | cut -d' ' -f1)
    file=$(echo "$entry" | cut -d' ' -f2)
    proj_name="FedProx_${name}_bcd"
    echo ""
    echo ">>> Running $proj_name ..."
    python -m fed_cd.federated.fed_main \
        --partition_json "$file" \
        --project_name "$proj_name" \
        --fedprox_mu "$FEDPROX_MU" \
        "${COMMON_ARGS[@]}"
done

echo ""
echo "========== All BCD experiments complete =========="
