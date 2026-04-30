#!/bin/bash
#SBATCH --job-name=pkcast_cfm
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --output=/home1/ppatel2025/pkcast/logs/cfm_%j.out
#SBATCH --error=/home1/ppatel2025/pkcast/logs/cfm_%j.err

# ── Activate environment ───────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nowcasting

# ── Paths (edit if your data lives elsewhere) ──────────────────────────────────
TRAIN_FILE="datasets/sevir/data/sevir_latent_vae/nowcast_training_full.h5"
TRAIN_META="datasets/sevir/data/sevir_latent_vae/nowcast_training_full_META.csv"
VAL_FILE="datasets/sevir/data/sevir_latent_vae/nowcast_validation_full.h5"
VAL_META="datasets/sevir/data/sevir_latent_vae/nowcast_validation_full_META.csv"

PARTIAL_EVAL_FILE="datasets/sevir/data/sevir_full/nowcast_validation_full.h5"
PARTIAL_EVAL_META="datasets/sevir/data/sevir_full/nowcast_validation_full_META.csv"

# ── Launch ─────────────────────────────────────────────────────────────────────
cd /home1/ppatel2025/pkcast

mkdir -p slurm/logs

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    scripts/train_pkcast.py \
    --config configs/pkcast.yaml \
    --train_file            "$TRAIN_FILE" \
    --train_meta            "$TRAIN_META" \
    --val_file              "$VAL_FILE" \
    --val_meta              "$VAL_META" \
    --partial_evaluation_file "$PARTIAL_EVAL_FILE" \
    --partial_evaluation_meta "$PARTIAL_EVAL_META"
