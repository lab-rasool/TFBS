#!/bin/bash
# Phase 0 decision gate: build the heterogeneous zoo, train the gate, report whether
# the per-TF ORACLE over the zoo can clear DNABERT's 0.749 OOD AUC.
# Usage: sbatch slurm/phase0.sbatch "<backbones>" "<out_dir>" [seed]
#   e.g. sbatch slurm/phase0.sbatch "ConvNet,DeepSEA,DanQ" ./results/phase0_cnn      (fast CNN-only)
#        sbatch slurm/phase0.sbatch "ConvNet,DeepSEA,DanQ,DNABERT6" ./results/phase0 (full zoo)
#SBATCH --job-name=hetmoe-p0
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/proj/rasool_lab_projects/Aakash/TFBS/slurm/logs/phase0_%j.log
#SBATCH --error=/proj/rasool_lab_projects/Aakash/TFBS/slurm/logs/phase0_%j.log

set -x
PROJ="${SLURM_SUBMIT_DIR:-/proj/rasool_lab_projects/Aakash/TFBS}"
PYTHON="${PYTHON:-/home/4483031/miniconda3/bin/python}"
cd "$PROJ"
export HF_HOME="$PROJ/.hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PROJ" TF_CPP_MIN_LOG_LEVEL=3

BACKBONES="${1:-ConvNet,DeepSEA,DanQ,DNABERT6}"
OUT="${2:-./results/phase0}"
SEED="${3:-42}"

nvidia-smi -L
echo "backbones=$BACKBONES out=$OUT seed=$SEED"
"$PYTHON" -m experiments.hetmoe.decision_gate --seed "$SEED" --backbones "$BACKBONES" --out "$OUT"
echo "EXIT_CODE=$?"
