#!/bin/bash
# Centralized baseline training (BIT-CD on WHU-GCD)
# Run from FedChange/ directory:
#   bash scripts/run_centralized.sh

EPOCHS=200
BATCH_SIZE=8
LR=0.01
IMG_SIZE=256

# Strategy A: gcd + ugcd_full (3:1 ratio)
python -m fed_cd.centralized_main \
    --mode centralized \
    --data_root ../WHU-GCD \
    --train_sources "gcd,ugcd_full" \
    --net_G base_transformer_pos_s4_dd8 \
    --num_classes 2 \
    --img_size $IMG_SIZE \
    --epochs $EPOCHS \
    --local_bs $BATCH_SIZE \
    --lr $LR \
    --lr_policy linear \
    --optimizer sgd \
    --pretrained True \
    --eval_splits "val,test,test2" \
    --global_test_frequency 20 \
    --save_frequency 20 \
    --project_name "Centralized_A_gcd_ugcdfull" \
    --checkpoint_root "results" \
    --seed 42

# Strategy B: gcd + ucd + ugcd (6:1:1 ratio)
python -m fed_cd.centralized_main \
    --mode centralized \
    --data_root ../WHU-GCD \
    --train_sources "gcd,ucd,ugcd" \
    --net_G base_transformer_pos_s4_dd8 \
    --num_classes 2 \
    --img_size $IMG_SIZE \
    --epochs $EPOCHS \
    --local_bs $BATCH_SIZE \
    --lr $LR \
    --lr_policy linear \
    --optimizer sgd \
    --pretrained True \
    --eval_splits "val,test,test2" \
    --global_test_frequency 20 \
    --save_frequency 20 \
    --project_name "Centralized_B_gcd_ucd_ugcd" \
    --checkpoint_root "results" \
    --seed 42
