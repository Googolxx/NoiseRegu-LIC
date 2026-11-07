#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="checkpoints/dmci_base_mse_noise.pth"
IMAGE_DIR="/path/to/kodak"
MODEL="dmci"  # dmci, dmci_medium, or dmci_large
METRICS="psnr"  # Use psnr,ms_ssim,lpips,dists for perceptual evaluation
OUTPUT="results/evaluations/dmci_base_noise_kodak.json"
SAVE_DIR=""  # Optional: save .bin streams and .png reconstructions
GPU=0
PYTHON=python

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"

save_args=()
if [[ -n "$SAVE_DIR" ]]; then
    save_args=(--save_dir "$SAVE_DIR")
fi

mkdir -p "$(dirname -- "$OUTPUT")"
"$PYTHON" evaluate.py "$CHECKPOINT" "$IMAGE_DIR" \
    --model "$MODEL" --metrics "$METRICS" --output "$OUTPUT" "${save_args[@]}"
