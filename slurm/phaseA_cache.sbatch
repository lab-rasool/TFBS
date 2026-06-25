#!/bin/bash
# Phase A: build the full heterogeneous zoo (ConvNet + DeepSEA + DanQ + fine-tuned
# DNABERT-6, per TF) and cache every expert's E-dim embedding + the DNABERT-6 baseline
# predictions. The one GPU-heavy step (3 DNABERT-6 fine-tunes dominate).
# Usage: sbatch slurm/phaseA_cache.sbatch [seed]
#SBATCH --job-name=hetmoe-A
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/proj/rasool_lab_projects/Aakash/TFBS/slurm/logs/phaseA_%j.log
#SBATCH --error=/proj/rasool_lab_projects/Aakash/TFBS/slurm/logs/phaseA_%j.log

set -x
PROJ="${SLURM_SUBMIT_DIR:-/proj/rasool_lab_projects/Aakash/TFBS}"
PYTHON="${PYTHON:-/home/4483031/miniconda3/bin/python}"
cd "$PROJ"
export HF_HOME="$PROJ/.hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PROJ" TF_CPP_MIN_LOG_LEVEL=3
SEED="${1:-42}"
nvidia-smi -L
"$PYTHON" -m experiments.hetmoe.cache_embeddings --seed "$SEED" \
    --backbones ConvNet,DeepSEA,DanQ,DNABERT6 --dnabert_epochs 3
echo "EXIT_CODE=$?"
