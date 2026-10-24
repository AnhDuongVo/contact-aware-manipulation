#!/usr/bin/env bash
# Example SLURM submission for a CPU node (adapt partition/account to your cluster).
#   sbatch scripts/slurm_job.sh
#SBATCH --job-name=cambench
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=1
CONFIG=${CONFIG:-configs/full.yaml} WORKERS=$SLURM_CPUS_PER_TASK bash scripts/run_full.sh
