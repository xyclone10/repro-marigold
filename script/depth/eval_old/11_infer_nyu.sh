#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value
ckpt=${1:-"prs-eth/marigold-depth-v1-0"}
# ckpt=${1:-"/home/shania/Marigold/output/train_marigold_depth_h90v10/checkpoint/latest"}
subfolder=${2:-"eval"}

python script/depth/infer.py \
    --checkpoint $ckpt \
    --seed 1234 \
    --base_data_dir $BASE_DATA_DIR \
    --denoise_steps 1 \
    --ensemble_size 10 \
    --processing_res 0 \
    --dataset_config config/dataset_depth/data_nyu_test.yaml \
    --output_dir output/${subfolder}/nyu_test_d1/prediction
