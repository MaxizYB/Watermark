#!/usr/bin/env bash
# Quick smoke test: 10 images, clean + jpeg_50, both methods
set -e
cd "$(dirname "$0")/.."

MODEL="/home/monyon/.cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c"
N=10

echo "=== Smoke test: FDW clean ==="
python run_fdw.py --method fdw --num $N --attack clean --model_path $MODEL

echo "=== Smoke test: GS clean ==="
python run_fdw.py --method gs --num $N --attack clean --model_path $MODEL

echo "=== Smoke test: FDW vs GS (jpeg_50) ==="
python run_fdw.py --method both --num $N --attack jpeg_50 --model_path $MODEL

echo "Done. Results in ./output/"
