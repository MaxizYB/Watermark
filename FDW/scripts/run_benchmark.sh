#!/usr/bin/env bash
# Full benchmark: all attack groups, 100 images, FDW vs GS
set -e
cd "$(dirname "$0")/.."

MODEL="/home/monyon/.cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c"
N=100
OUT="./benchmark_output"

echo "=== Full attack benchmark: FDW vs GS, N=$N ==="
python run_attack_benchmark.py \
    --method both \
    --num $N \
    --model_path $MODEL \
    --output_path $OUT \
    --plot

echo "Done. Results in $OUT/"
