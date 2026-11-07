#!/usr/bin/env bash
set -euo pipefail

DATASET="kodak"
DATASET_DIR="/path/to/kodak"
VTM_BIN_DIR="/path/to/vtm-22.0/bin"
PERCEPTUAL=false  # false: PSNR; true: PSNR, MS-SSIM, LPIPS, and DISTS
SKIP_ENCODE=false # Perceptual mode only: reuse existing reconstructions

MSE_QPS="27,32,37,42,47,52"
PERCEPTUAL_QPS="32,36,40,44,48,52,56,60"
NJOBS=8
DEVICE="cuda:0"
PYTHON=python

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$PERCEPTUAL" == "true" ]]; then
    OUTPUT_JSON="results/evaluations/vtm_perceptual_${DATASET}.json"
    RECON_DIR="recon/vtm_perceptual/vtm_${DATASET}"
    mkdir -p "$(dirname -- "$OUTPUT_JSON")" "$RECON_DIR"

    skip_args=()
    if [[ "$SKIP_ENCODE" == "true" ]]; then
        skip_args=(--skip_encode)
    fi

    "$PYTHON" test_vtm_perceptual.py "$DATASET_DIR" \
        -q "$PERCEPTUAL_QPS" -b "$VTM_BIN_DIR" \
        -c bench/vtm/encoder_intra_vtm.cfg -j "$NJOBS" \
        --device "$DEVICE" --recon_dir "$RECON_DIR" \
        --output_json "$OUTPUT_JSON" "${skip_args[@]}"
else
    RAW_JSON="recon/vtm_raw/vtm_${DATASET}.json"
    OUTPUT_JSON="results/evaluations/vtm_${DATASET}.json"
    mkdir -p "$(dirname -- "$RAW_JSON")" "$(dirname -- "$OUTPUT_JSON")"

    "$PYTHON" -m bench vtm "$DATASET_DIR" \
        -q "$MSE_QPS" -b "$VTM_BIN_DIR" \
        -c bench/vtm/encoder_intra_vtm.cfg -j "$NJOBS" > "$RAW_JSON"
    "$PYTHON" tools/vtm.py "$RAW_JSON" "$OUTPUT_JSON" "$DATASET" --qps "$MSE_QPS"
fi
