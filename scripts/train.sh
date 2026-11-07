#!/usr/bin/env bash
set -euo pipefail

# Training settings
DATASET="/path/to/train"
VAL_DIR="/path/to/val"
SAVE_DIR="outputs/dmci_base_noise"
CHECKPOINT=""  # Optional initialization checkpoint
MODEL="dmci"   # dmci, dmci_medium, or dmci_large
INPUT_NOISE=true
SIGMA=0.5

# Hardware settings
GPUS="0,1,2,3"
NPROC=4
MASTER_PORT=29600
PYTHON=python

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES="$GPUS"

noise_args=()
if [[ "$INPUT_NOISE" == "true" ]]; then
    noise_args=(--input_noise --noise_sigma_start "$SIGMA")
fi

checkpoint_args=()
if [[ -n "$CHECKPOINT" ]]; then
    checkpoint_args=(--checkpoint "$CHECKPOINT")
fi

# Stage 1: 256x256, with optional input noise
"$PYTHON" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    train.py -d "$DATASET" --val_dir "$VAL_DIR" --save_dir "$SAVE_DIR" \
    --model "$MODEL" --epochs 480 --lr_epoch 410 --patch-size 256 256 \
    --batch-size 8 --learning-rate 1e-4 "${checkpoint_args[@]}" "${noise_args[@]}"

# Stage 2: 512x512 fine-tuning
"$PYTHON" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    train.py -d "$DATASET" --val_dir "$VAL_DIR" --save_dir "$SAVE_DIR/512" \
    --model "$MODEL" --epochs 22 --lr_epoch 18 --patch-size 512 512 \
    --batch-size 4 --learning-rate 1e-5 --checkpoint "$SAVE_DIR/checkpoint_latest.pth.tar"
