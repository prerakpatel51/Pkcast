#!/bin/bash
#SBATCH --job-name=pkcast_vae
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --output=/home1/ppatel2025/pkcast/logs/vae_%j.out
#SBATCH --error=/home1/ppatel2025/pkcast/logs/vae_%j.err

# ── Activate environment ───────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nowcasting

# ── Paths (edit if your data lives elsewhere) ──────────────────────────────────
TRAIN_FILE="datasets/sevir/data/sevir_full/nowcast_training_full.h5"
TRAIN_META="datasets/sevir/data/sevir_full/nowcast_training_full_META.csv"
VAL_FILE="datasets/sevir/data/sevir_full/nowcast_validation_full.h5"
VAL_META="datasets/sevir/data/sevir_full/nowcast_validation_full_META.csv"

# ── Launch ─────────────────────────────────────────────────────────────────────
export TMPDIR=/home1/ppatel2025/tmp
export TORCHELASTIC_LOG_DIR=/home1/ppatel2025/tmp
export TORCHINDUCTOR_CACHE_DIR=/home1/ppatel2025/tmp/torchinductor
export TORCH_HOME=/home1/ppatel2025/tmp/torch_home

mkdir -p /home1/ppatel2025/tmp/torchinductor /home1/ppatel2025/tmp/torch_home

cd /home1/ppatel2025/pkcast

mkdir -p slurm/logs

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    scripts/train_autoencoder.py \
    --config configs/autoencoder.yaml \
    --train_file "$TRAIN_FILE" \
    --train_meta "$TRAIN_META" \
    --val_file   "$VAL_FILE" \
    --val_meta   "$VAL_META"
