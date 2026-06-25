#!/bin/bash
# Phase B+C: config sweep over the cached zoo + pre-registered selection (in-dist only)
# + publication-grade full evaluation (bootstrap, paired vs DNABERT, TOST, calibration).
# Cheap (reuses Phase-A cache; no expert forward). Submit chained:
#   sbatch --dependency=afterok:<phaseA_jobid> slurm/phaseBC.sbatch [seed]
#SBATCH --job-name=hetmoe-BC
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/proj/rasool_lab_projects/Aakash/TFBS/slurm/logs/phaseBC_%j.log
#SBATCH --error=/proj/rasool_lab_projects/Aakash/TFBS/slurm/logs/phaseBC_%j.log

set -x
PROJ="${SLURM_SUBMIT_DIR:-/proj/rasool_lab_projects/Aakash/TFBS}"
PYTHON="${PYTHON:-/home/4483031/miniconda3/bin/python}"
cd "$PROJ"
export HF_HOME="$PROJ/.hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PROJ" TF_CPP_MIN_LOG_LEVEL=3
SEED="${1:-42}"
"$PYTHON" -m experiments.hetmoe.sweep --seed "$SEED" --n_boot 1000
echo "EXIT_CODE=$?"
