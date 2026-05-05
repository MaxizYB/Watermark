#!/usr/bin/env bash
# Sweep lambda_freq and alpha_max to find best hyperparameters
set -e
cd "$(dirname "$0")/.."

MODEL="/home/monyon/.cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c"
N=30

for LAMBDA in 0.04 0.08 0.12 0.16; do
    for ALPHA in 0.005 0.010 0.015 0.020; do
        TAG="lambda${LAMBDA}_alpha${ALPHA}"
        echo "--- $TAG ---"
        python run_fdw.py --method fdw --num $N \
            --attack jpeg_50 \
            --lambda_freq $LAMBDA \
            --alpha_max $ALPHA \
            --output_path "./hyperparam_output/$TAG" \
            --model_path $MODEL
    done
done

echo "Hyperparam sweep done. Results in ./hyperparam_output/"
