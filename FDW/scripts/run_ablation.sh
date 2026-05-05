#!/usr/bin/env bash
# Ablation: test individual FDW components
# Compares: GS | FDW-noECC-noFDSC | FDW-noFDSC | FDW-full
set -e
cd "$(dirname "$0")/.."

MODEL="/home/monyon/.cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c"
N=50
ATTACKS="clean jpeg_50 jpeg_25 crop_060 rotate_45 gauss_noise_005 stirmark_all"

run_single() {
    local TAG=$1
    shift
    echo "--- Ablation: $TAG ---"
    for ATK in $ATTACKS; do
        python run_fdw.py --method fdw --num $N --attack $ATK \
            --output_path "./ablation_output/$TAG" \
            --model_path $MODEL "$@"
    done
}

# GS baseline
for ATK in $ATTACKS; do
    python run_fdw.py --method gs --num $N --attack $ATK \
        --output_path "./ablation_output/gs" --model_path $MODEL
done

# FDW: no ECC, no FDSC (only freq init noise)
run_single "fdw_freq_only" --no_ecc --no_fdsc --no_fd_detect

# FDW: freq init + ECC, no FDSC
run_single "fdw_freq_ecc" --use_ecc --no_fdsc

# FDW: full (freq init + ECC + FDSC + dual detect)
run_single "fdw_full" --use_ecc --use_fdsc --use_fd_detect

echo "Ablation done. Results in ./ablation_output/"
