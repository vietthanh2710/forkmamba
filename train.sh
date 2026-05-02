#!/bin/bash

# ForkMamba Training Script
# Modify the variables below according to your setup

# ==================== CONFIGURATION ====================
now=$(date +"%Y%m%d_%H%M%S")

dataset='pascal'
method='unimatch'
exp='r101'
split='732'

num_gpus=2
port=12345
data_path="/path/to/your/data.npy"
save_path="/path/to/checkpoints"
config_path="config.yaml"

# ==================== TRAINING ====================
mkdir -p $save_path

echo "Starting ForkMamba training..."
echo "Number of GPUs: $num_gpus"
echo "Port: $port"
echo "Data path: $data_path"
echo "Save path: $save_path"
echo "Config path: $config_path"
echo "Log file: $save_path/training_$now.log"
echo "=========================================="

# Run distributed training
python -m torch.distributed.launch \
    --nproc_per_node=$num_gpus \
    --master_addr=localhost \
    --master_port=$port \
    train.py \
    --config=$config_path \
    --data-path=$data_path \
    --save-path=$save_path \
    --port=$port 2>&1 | tee $save_path/training_$now.log
