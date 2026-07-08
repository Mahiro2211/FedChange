#!/bin/bash
# Centralized BCD experiment: the performance upper bound for federated runs.
# Trains a single BIT-CD model on the union of all training samples
# (gcd + ugcd_full + ucd + ugd), with the SAME hyperparameters as the federated
# BIT-CD experiments in run_fed_bcd.sh — the only difference is centralized vs.
# federated optimization, so the two are directly comparable.
#
# Run from FedChange/ directory:
#   bash scripts/run_centralized.sh
# Positional args (epochs batch_size img_size lr):
#   bash scripts/run_centralized.sh 200 8 256 0.01

EPOCHS=${1:-200}
BATCH_SIZE=${2:-8}
IMG_SIZE=${3:-256}
LR=${4:-0.01}

# Net_G model (BIT-CD variants or torchange baselines). Default: framework BIT-CD.
NET_G=${5:-base_transformer_pos_s4_dd8}
# Suffix baked into project_name when sweeping multiple models.
TAG=${6:-base}
SEED=${7:-42}

PROJ_NAME="Centr_${TAG}_bcd_s${SEED}"

echo ""
echo "========== Centralized BCD Experiment =========="
echo ">>> Model: $NET_G"
echo ">>> Project: $PROJ_NAME"
echo ""

python -m fed_cd.centralized.cen_main \
    --data_root "../WHU-GCD" \
    --net_G "$NET_G" \
    --num_classes 2 \
    --img_size "$IMG_SIZE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --lr_policy linear \
    --optimizer sgd \
    --pretrained True \
    --eval_splits val,test,test2 \
    --global_test_frequency 20 \
    --save_frequency 20 \
    --checkpoint_root results/centralized \
    --project_name "$PROJ_NAME" \
    --seed "$SEED"

echo ""
echo "========== Centralized BCD experiment complete =========="
