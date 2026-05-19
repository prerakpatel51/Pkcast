#!/bin/bash
#SBATCH --job-name=pkcast_more_pairs
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home1/ppatel2025/pkcast/logs/infer_more_pairs_%j.out
#SBATCH --error=/home1/ppatel2025/pkcast/logs/infer_more_pairs_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nowcasting

export TMPDIR=/home1/ppatel2025/tmp
export TORCHINDUCTOR_CACHE_DIR=/home1/ppatel2025/tmp/torchinductor
export TORCH_HOME=/home1/ppatel2025/tmp/torch_home

mkdir -p /home1/ppatel2025/tmp/torchinductor /home1/ppatel2025/tmp/torch_home
mkdir -p /home1/ppatel2025/pkcast/logs

cd /home1/ppatel2025/pkcast

OUTPUT_DIR="artifacts/sevir/inference_more_pairs"
README_ASSETS_DIR="readme_assets"
mkdir -p "${OUTPUT_DIR}" "${README_ASSETS_DIR}"

SAMPLE_INDICES=(${SAMPLE_INDICES:-50 75 100 125})
README_SAMPLE_NUMBER=3

for SAMPLE_INDEX in "${SAMPLE_INDICES[@]}"; do
    printf '\n=== Running sample %s as README sample %s ===\n' "${SAMPLE_INDEX}" "${README_SAMPLE_NUMBER}"

    python scripts/inference.py \
        --config configs/pkcast.yaml \
        --cfm_checkpoint artifacts/sevir/pkcast/2026-04-22_11-20-07_cfm_pkcast_ddp_14b2d14f_main/models/early_stopping_model.pt \
        --test_file datasets/sevir/data/sevir_full/nowcast_testing_full.h5 \
        --test_meta datasets/sevir/data/sevir_full/nowcast_testing_full_META.csv \
        --sample_index "${SAMPLE_INDEX}" \
        --probabilistic_samples 1 \
        --output_dir "${OUTPUT_DIR}"

    printf -v PADDED_INDEX "%05d" "${SAMPLE_INDEX}"
    cp "${OUTPUT_DIR}/sample_${PADDED_INDEX}_target.gif" \
        "${README_ASSETS_DIR}/pkcast_sample${README_SAMPLE_NUMBER}_ground_truth.gif"
    cp "${OUTPUT_DIR}/sample_${PADDED_INDEX}_prediction.gif" \
        "${README_ASSETS_DIR}/pkcast_sample${README_SAMPLE_NUMBER}_prediction.gif"

    README_SAMPLE_NUMBER=$((README_SAMPLE_NUMBER + 1))
done

printf '\nSaved raw inference outputs under %s\n' "${OUTPUT_DIR}"
printf 'Saved README-format GIF pairs under %s/pkcast_sample{3..%s}_*.gif\n' \
    "${README_ASSETS_DIR}" "$((README_SAMPLE_NUMBER - 1))"
