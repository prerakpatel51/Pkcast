#!/bin/bash
#SBATCH --job-name=pkcast_infer_gif
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home1/ppatel2025/pkcast/logs/infer_gif_%j.out
#SBATCH --error=/home1/ppatel2025/pkcast/logs/infer_gif_%j.err

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nowcasting

export TMPDIR=/home1/ppatel2025/tmp
export TORCHINDUCTOR_CACHE_DIR=/home1/ppatel2025/tmp/torchinductor
export TORCH_HOME=/home1/ppatel2025/tmp/torch_home

mkdir -p /home1/ppatel2025/tmp/torchinductor /home1/ppatel2025/tmp/torch_home
mkdir -p /home1/ppatel2025/pkcast/logs

cd /home1/ppatel2025/pkcast

python scripts/inference.py \
    --config configs/pkcast.yaml \
    --cfm_checkpoint artifacts/sevir/pkcast/2026-04-22_11-20-07_cfm_pkcast_ddp_14b2d14f_main/models/early_stopping_model.pt \
    --test_file datasets/sevir/data/sevir_full/nowcast_testing_full.h5 \
    --test_meta datasets/sevir/data/sevir_full/nowcast_testing_full_META.csv \
    --sample_index 0 \
    --probabilistic_samples 1 \
    --output_dir artifacts/sevir/inference_gif
