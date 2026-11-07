#!/usr/bin/env bash
set -euo pipefail

REFERENCE="results/dmci_base_clean_kodak.json"
TEST="results/dmci_base_noise_kodak.json"
METRIC="psnr"  # psnr, ms_ssim, lpips, dists, or all
OUTPUT="results/bdrate/dmci_base_kodak.json"  # Leave empty to print without saving
PYTHON=python

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

args=("$REFERENCE" "$TEST" --metric "$METRIC")
if [[ -n "$OUTPUT" ]]; then
    mkdir -p "$(dirname -- "$OUTPUT")"
    args+=(--output "$OUTPUT")
fi

"$PYTHON" tools/bdrate.py "${args[@]}"
